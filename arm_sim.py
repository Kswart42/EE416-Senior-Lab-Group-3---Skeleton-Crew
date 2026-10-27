"""
arm_sim.py - live 3D torso + arm simulation from two IMUs and the distance
between the boards.

Hardware assumption
-------------------
  imu1 (access point) : on the TORSO (chest / sternum)
  imu2 (joiner)       : on the FOREARM, just above the wrist

What each input is used for
---------------------------
  imu1 roll/pitch/yaw  -> torso orientation
  imu2 roll/pitch/yaw  -> forearm direction (relative to the torso)
  distance imu1<->imu2 -> elbow bend. The shoulder is fixed on the torso and
                          the wrist sensor is a known forearm-length away from
                          the elbow, so the measured distance pins down how far
                          the elbow is bent (law of cosines, see solve()).

Run
---
  python arm_sim.py            # live, listens for the ESP32 UDP packets
  python arm_sim.py --sim      # no hardware: synthetic IMUs + distance demo

Do not run it together with receiver_ranging.py (both want UDP port 4210).

Calibration (do this every session; keys are pressed in the plot window)
------------------------------------------------------------------------
  1. Stand straight, arm hanging relaxed at your side, facing forward. Hold still.
     Press  N
  2. Hold the arm straight out in front of you, horizontal, palm in. Hold still.
     Press  F      (press F again any time to re-zero the heading drift)
  D toggles using the distance (falls back to a fixed elbow bend when off).
  Q quits.

Why two poses: the sensors have no magnetometer, so each board's yaw starts at
0 in its own arbitrary direction and slowly drifts. Pose N learns the forearm
axis in the sensor's own frame (from gravity); pose F learns the heading offset
between the two boards.

Packet format (from main.cpp): id,device_ms,ax,ay,az,gx,gy,gz,roll,pitch,yaw,rssi
"""
import argparse
import math
import socket
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------- settings
PORT = 4210

# Same radio -> distance model as receiver_ranging.py. Recalibrate for your boards.
RSSI_AT_1M = -45.0
PATH_LOSS_EXPONENT = 2.5

# Body dimensions in metres. MEASURE YOURSELF, these matter for the elbow solve.
SHOULDER_HALF_WIDTH = 0.19   # chest sensor to shoulder joint, sideways
SHOULDER_UP = 0.12           # chest sensor to shoulder joint, upwards
SHOULDER_BACK = 0.02         # shoulder joint is slightly behind the sternum
UPPER_ARM = 0.30             # shoulder to elbow
FOREARM = 0.26               # elbow to the wrist sensor
HAND = 0.09                  # wrist sensor to fingertips (drawing only)
HEAD_UP = 0.33
NECK_UP = 0.20
HIP_DOWN = 0.50
HIP_HALF_WIDTH = 0.10

# How the torso board sits on you:
#   "flat"  : board x forward, y left, z up (e.g. on the upper back / a strap)
#   "chest" : board lying vertically on the sternum, z pointing out of the chest,
#             y pointing up, x pointing to your left
MOUNTS = {
    "flat": np.eye(3),
    "chest": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
}

ELBOW_DEFAULT_DEG = 25.0     # elbow bend used when no distance is available
# Where the elbow "wants" to point if several poses fit the data (body frame:
# x forward, y left, z up): down and slightly back, like a relaxed elbow.
ELBOW_PREF = np.array([-0.3, 0.0, -1.0])
ELBOW_PREF /= np.linalg.norm(ELBOW_PREF)

ELBOW_SMOOTH = 0.3           # 0..1, higher = snappier but jitterier elbow
STALE_SEC = 1.0              # no packet for this long = device considered gone
TRAIL_POINTS = 40


# ------------------------------------------------------------------- maths
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_R(roll_deg, pitch_deg, yaw_deg):
    """Sensor->world rotation. Matches main.cpp: z up, ZYX (yaw, pitch, roll)."""
    return (rot_z(math.radians(yaw_deg)) @ rot_y(math.radians(pitch_deg))
            @ rot_x(math.radians(roll_deg)))


def R_to_euler_deg(R):
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def heading(v):
    return math.atan2(v[1], v[0])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def cone_pick(axis, c, pref):
    """Unit vector u with u.axis == c that is closest to the direction `pref`.

    All such vectors form a cone around `axis`; the position around the cone is
    the one thing the sensors cannot tell us, so we choose the most natural one.
    """
    c = max(-1.0, min(1.0, c))
    s = math.sqrt(1.0 - c * c)
    perp = pref - np.dot(pref, axis) * axis
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(axis, [0.0, 1.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(axis, [1.0, 0.0, 0.0])
    return c * axis + s * unit(perp)


def hinge_pick(axis, c, h, f_hat):
    """Like cone_pick, but also require u to be perpendicular to the hinge axis h.

    The elbow is a hinge: upper arm and forearm both lie in the plane
    perpendicular to its axis. That plane cuts the distance cone in (at most)
    two points, which removes the swivel ambiguity. Of the two, the elbow only
    bends one way (no hyperextension), so the correct one is the one whose
    (upper arm x forearm) points against the hinge axis. Returns None if degenerate.
    """
    e1 = np.cross(h, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(h, [1.0, 0.0, 0.0])
    e1 = unit(e1)
    e2 = np.cross(h, e1)
    p = np.array([np.dot(axis, e1), np.dot(axis, e2)])
    pn = float(np.linalg.norm(p))
    if pn < 0.05:
        return None
    k = max(-1.0, min(1.0, c / pn))        # clamp: noisy distance can overshoot
    ph = p / pn
    pperp = np.array([-ph[1], ph[0]])
    r = math.sqrt(1.0 - k * k)
    cands = []
    for sgn in (1.0, -1.0):
        xy = k * ph + sgn * r * pperp
        cands.append(xy[0] * e1 + xy[1] * e2)
    return min(cands, key=lambda u: float(np.dot(np.cross(u, f_hat), h)))


def rssi_to_distance(rssi):
    return 10 ** ((RSSI_AT_1M - rssi) / (10.0 * PATH_LOSS_EXPONENT))


# -------------------------------------------------------- shared live state
class RangeFilter:
    """Median over the last few samples (kills spikes), then a slow EMA."""

    def __init__(self, n=7, alpha=0.4):
        self.buf = deque(maxlen=n)
        self.alpha = alpha
        self.val = None

    def add(self, d):
        self.buf.append(d)
        m = float(np.median(self.buf))
        self.val = m if self.val is None else self.val + self.alpha * (m - self.val)

    def reset(self):
        self.buf.clear()
        self.val = None


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.R = {1: None, 2: None}
        self.seen = {1: 0.0, 2: 0.0}
        self.range = RangeFilter()
        self.range_seen = 0.0
        self.rssi = 0.0
        self.counts = {}

    def update(self, dev, roll, pitch, yaw, rssi=0.0, now=None):
        if dev not in (1, 2):
            return
        now = time.time() if now is None else now
        R = euler_to_R(roll, pitch, yaw)
        with self.lock:
            self.R[dev] = R
            self.seen[dev] = now
            self.counts[dev] = self.counts.get(dev, 0) + 1
            if dev == 2:
                if rssi != 0.0:
                    self.range.add(rssi_to_distance(rssi))
                    self.rssi = rssi
                    self.range_seen = now
                else:
                    self.range.reset()

    def snapshot(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            fresh = {k: self.R[k] is not None and now - self.seen[k] < STALE_SEC
                     for k in (1, 2)}
            dist = self.range.val if now - self.range_seen < STALE_SEC else None
            return self.R[1], self.R[2], fresh, dist, self.rssi


def receiver_loop(state, stop):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(1.0)
    print(f"Listening on UDP port {PORT} ...")
    seen = set()
    while not stop.is_set():
        try:
            data, addr = sock.recvfrom(512)
        except socket.timeout:
            continue
        try:
            p = data.decode().strip().split(",")
            if len(p) not in (11, 12):
                continue
            dev = int(p[0])
            roll, pitch, yaw = float(p[8]), float(p[9]), float(p[10])
            rssi = float(p[11]) if len(p) == 12 else 0.0
        except (UnicodeDecodeError, ValueError):
            continue
        if dev not in seen:
            seen.add(dev)
            print(f"New device: id={dev} from {addr[0]}")
        state.update(dev, roll, pitch, yaw, rssi)
    sock.close()


# ------------------------------------------------------------ the arm model
class ArmTracker:
    """Turns (torso orientation, forearm orientation, distance) into joint positions.

    Body frame: x forward, y left, z up, origin at the torso sensor.
    """

    def __init__(self, side="right", mount="flat", elbow_mode="hinge"):
        self.side = -1.0 if side == "right" else 1.0   # +y is the body's left
        self.M = MOUNTS[mount]
        self.elbow_mode = elbow_mode   # "hinge" or "cone" (see solve)
        self.hs = None                 # elbow hinge axis in the forearm board's frame
        self.shoulder = np.array([-SHOULDER_BACK,
                                  self.side * SHOULDER_HALF_WIDTH,
                                  SHOULDER_UP])
        self.v = None        # forearm axis in the forearm board's own frame
        self.E = np.eye(3)   # small torso-mount correction learned at pose N
        self.h0 = 0.0        # torso heading at pose N (so the figure starts facing +x)
        self.psi = None      # heading offset: imu2 world -> imu1 world
        self.u_s = None      # smoothed upper-arm direction

    # ---- calibration -------------------------------------------------
    def capture_neutral(self, Rt, Ra):
        self.v = Ra.T @ np.array([0.0, 0.0, -1.0])      # arm hangs along gravity
        Rwb0 = Rt @ self.M.T
        self.h0 = heading(Rwb0 @ np.array([1.0, 0.0, 0.0]))
        self.E = Rwb0.T @ rot_z(self.h0)                # make the torso exactly upright
        self.psi = None
        self.hs = None
        self.u_s = None

    def capture_forward(self, Rt, Ra):
        if self.v is None:
            return False
        f2 = Ra @ self.v
        if math.hypot(f2[0], f2[1]) < 0.5:              # arm not near horizontal
            return False
        fwd = self._true_R(Rt) @ np.array([1.0, 0.0, 0.0])
        self.psi = heading(fwd) - heading(f2)
        # Arm straight forward, thumb up: the elbow hinge axis is the body's y axis.
        h_world2 = rot_z(-self.psi) @ (self._true_R(Rt) @ np.array([0.0, 1.0, 0.0]))
        self.hs = Ra.T @ h_world2
        self.u_s = None
        return True

    @property
    def calibrated(self):
        return self.v is not None and self.psi is not None

    # ---- per-frame solve ---------------------------------------------
    def _true_R(self, Rt):
        return Rt @ self.M.T @ self.E

    def display_R(self, Rt):
        return rot_z(-self.h0) @ self._true_R(Rt)

    def solve(self, Rt, Ra, dist):
        Rtrue = self._true_R(Rt)
        f_world1 = rot_z(self.psi or 0.0) @ (Ra @ self.v)
        f_hat = unit(Rtrue.T @ f_world1)                # forearm direction, body frame
        F = FOREARM * f_hat
        S = self.shoulder

        # The wrist sensor sits at S + U + F, the torso sensor at the origin, so
        #   |U + A| = dist  with  A = S + F   and   |U| = UPPER_ARM.
        # Expanding gives the angle between U and A; U then lies on a cone around
        # A and we take the point on that cone nearest the relaxed-elbow pose.
        A = S + F
        a = float(np.linalg.norm(A))
        h = unit(Rtrue.T @ (rot_z(self.psi or 0.0) @ (Ra @ self.hs))) \
            if self.hs is not None else None
        if dist is not None and a > 1e-3:
            c = (dist ** 2 - UPPER_ARM ** 2 - a ** 2) / (2 * UPPER_ARM * a)
            u = None
            if self.elbow_mode == "hinge" and h is not None:
                u = hinge_pick(A / a, c, h, f_hat)
            if u is None:
                u = cone_pick(A / a, c, ELBOW_PREF)
            mode = "distance + " + ("hinge" if self.elbow_mode == "hinge" else "cone")
        else:
            u = cone_pick(f_hat, math.cos(math.radians(ELBOW_DEFAULT_DEG)), ELBOW_PREF)
            mode = "fixed elbow"

        self.u_s = u if self.u_s is None else unit(self.u_s + ELBOW_SMOOTH * (u - self.u_s))
        u = self.u_s
        U = UPPER_ARM * u
        elbow = S + U
        wrist = elbow + F
        hand = wrist + HAND * f_hat
        return {
            "S": S, "elbow": elbow, "wrist": wrist, "hand": hand,
            "flex": math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, f_hat)))))),
            "shoulder_ang": math.degrees(math.acos(max(-1.0, min(1.0, float(-u[2]))))),
            "mode": mode,
            "Rdisp": rot_z(-self.h0) @ Rtrue,
        }

    def rest_pose(self):
        S = self.shoulder
        down = np.array([0.0, 0.0, -1.0])
        elbow = S + UPPER_ARM * down
        wrist = elbow + FOREARM * down
        return {"S": S, "elbow": elbow, "wrist": wrist, "hand": wrist + HAND * down,
                "flex": 0.0, "shoulder_ang": 0.0, "mode": "not calibrated",
                "Rdisp": np.eye(3)}


# --------------------------------------------------------- synthetic source
class Sim:
    """Fake wearer: known joint angles -> the IMU angles + RSSI the boards would send.

    Because we know the truth, this doubles as a test of the whole pipeline.
    Timeline: 0-4 s arm down (press N at ~3 s), 4-8 s arm forward (F at ~7 s),
    then free movement of the arm and torso.
    """

    def __init__(self, side="right", mount="flat", noise_db=0.5, seed=1, pron_deg=25.0):
        self.pron_deg = pron_deg
        self.rng = np.random.default_rng(seed)
        self.side = -1.0 if side == "right" else 1.0
        self.M = MOUNTS[mount]
        self.noise_db = noise_db
        self.Rerr = rot_x(math.radians(4)) @ rot_y(math.radians(-3))  # sloppy torso mount
        self.psi = math.radians(70)                  # imu2 boots facing a different way
        self.Ma = rot_x(math.radians(90)) @ rot_z(math.radians(25))   # arbitrary forearm mount
        self.h_torso = math.radians(35)
        self.shoulder = np.array([-SHOULDER_BACK, self.side * SHOULDER_HALF_WIDTH, SHOULDER_UP])

    @staticmethod
    def _smooth(x):
        x = max(0.0, min(1.0, x))
        return x * x * (3 - 2 * x)

    def truth(self, t):
        up = self._smooth((t - 4.0) / 1.0)                 # raise arm forward
        free = self._smooth((t - 8.0) / 2.0)               # then free motion
        flex = 90.0 * up * (1 - free) + free * (55 + 35 * math.sin(0.6 * t))
        elbow = free * max(0.0, 65 + 55 * math.sin(0.9 * t + 1.0))
        abd = free * (12 + 12 * math.sin(0.4 * t))
        roll = free * 6 * math.sin(0.3 * t)
        pitch = free * 9 * math.sin(0.25 * t)
        head = free * 12 * math.sin(0.2 * t)

        Rsh = rot_x(self.side * math.radians(abd)) @ rot_y(-math.radians(flex))
        pron = free * self.pron_deg * math.sin(0.7 * t + 0.5)   # forearm twist
        R_fore_body = Rsh @ rot_y(-math.radians(elbow)) @ rot_z(math.radians(pron))
        down = np.array([0.0, 0.0, -1.0])
        U = UPPER_ARM * (Rsh @ down)
        F = FOREARM * (R_fore_body @ down)
        R_tb = (rot_z(self.h_torso + math.radians(head)) @ rot_y(math.radians(pitch))
                @ rot_x(math.radians(roll)))
        return {"R_fore_body": R_fore_body, "R_tb": R_tb,
                "elbow": self.shoulder + U, "wrist": self.shoulder + U + F,
                "flex": elbow}

    def feed(self, state, t, now=None):
        g = self.truth(t)
        R_t = g["R_tb"] @ self.Rerr @ self.M
        R_a = rot_z(self.psi) @ g["R_tb"] @ g["R_fore_body"] @ self.Ma
        d = float(np.linalg.norm(g["wrist"]))
        rssi = (RSSI_AT_1M - 10 * PATH_LOSS_EXPONENT * math.log10(max(d, 0.02))
                + self.rng.normal(0, self.noise_db))
        state.update(1, *R_to_euler_deg(R_t), 0.0, now)
        state.update(2, *R_to_euler_deg(R_a), rssi, now)
        return g


def sim_loop(state, tracker, sim, stop):
    t0 = time.time()
    did_n = did_f = False
    while not stop.is_set():
        t = time.time() - t0
        sim.feed(state, t)
        if not did_n and t > 3.0:
            Rt, Ra, *_ = state.snapshot()
            tracker.capture_neutral(Rt, Ra)
            did_n = True
            print("[sim] neutral captured")
        if not did_f and t > 7.0:
            Rt, Ra, *_ = state.snapshot()
            did_f = tracker.capture_forward(Rt, Ra)
            print("[sim] forward captured")
        time.sleep(0.02)


# ------------------------------------------------------------------ viewer
def build_figure(side):
    plt.rcParams["keymap.fullscreen"] = []      # we use F for calibration
    fig = plt.figure(figsize=(9, 7.5))
    fig.canvas.manager.set_window_title("Torso + arm simulation")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(-0.65, 0.65)
    ax.set_zlim(-0.6, 0.5)
    ax.set_box_aspect((1.3, 1.3, 1.1))
    ax.set_xlabel("forward")
    ax.set_ylabel("left")
    ax.set_zlabel("up")
    ax.view_init(elev=15, azim=-35 if side == "right" else 35)

    art = {}
    (art["torso"],) = ax.plot([], [], [], "-", color="0.35", lw=3)
    (art["neck"],) = ax.plot([], [], [], "-", color="0.35", lw=3)
    (art["head"],) = ax.plot([], [], [], "o", color="0.35", ms=22)
    (art["other"],) = ax.plot([], [], [], "-", color="0.65", lw=5, solid_capstyle="round")
    (art["arm"],) = ax.plot([], [], [], "-o", color="tab:orange", lw=6, ms=8)
    (art["hand"],) = ax.plot([], [], [], "-", color="tab:red", lw=3)
    (art["trail"],) = ax.plot([], [], [], "-", color="tab:blue", lw=1, alpha=0.6)
    art["info"] = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top",
                            family="monospace", fontsize=9)
    return fig, ax, art


def set3d(line, pts):
    pts = np.array(pts)
    line.set_data(pts[:, 0], pts[:, 1])
    line.set_3d_properties(pts[:, 2])


def body_points(side):
    sgn = -1.0 if side == "right" else 1.0
    P = {
        "sh_a": np.array([-SHOULDER_BACK, sgn * SHOULDER_HALF_WIDTH, SHOULDER_UP]),
        "sh_b": np.array([-SHOULDER_BACK, -sgn * SHOULDER_HALF_WIDTH, SHOULDER_UP]),
        "hip_a": np.array([0.0, sgn * HIP_HALF_WIDTH, -HIP_DOWN]),
        "hip_b": np.array([0.0, -sgn * HIP_HALF_WIDTH, -HIP_DOWN]),
        "neck": np.array([-SHOULDER_BACK, 0.0, NECK_UP]),
        "head": np.array([0.0, 0.0, HEAD_UP]),
        "pelvis": np.array([0.0, 0.0, -HIP_DOWN]),
    }
    return P


def run_viewer(state, tracker, side, use_dist, sim_truth=None):
    fig, ax, art = build_figure(side)
    P = body_points(side)
    trail = deque(maxlen=TRAIL_POINTS)
    other_hand = P["sh_b"] + np.array([0.0, 0.0, -(UPPER_ARM + FOREARM)])

    def on_key(event):
        k = (event.key or "").lower()
        if k in ("n", "f"):
            Rt, Ra, fresh, _, _ = state.snapshot()
            if not (fresh[1] and fresh[2]):
                print("Need data from both boards before calibrating.")
                return
            if k == "n":
                tracker.capture_neutral(Rt, Ra)
                trail.clear()
                print("Neutral captured. Now raise the arm straight forward and press F.")
            elif tracker.capture_forward(Rt, Ra):
                print("Forward captured. Calibrated.")
            else:
                print("Forward not captured: press N first, and hold the arm horizontal.")
        elif k == "d":
            use_dist[0] = not use_dist[0]
            print("Use distance:", use_dist[0])

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        Rt, Ra, fresh, dist, rssi = state.snapshot()
        if not (fresh[1] and fresh[2]):
            art["info"].set_text(
                "Waiting for packets...\n"
                f"  imu1 (torso):   {'ok' if fresh[1] else 'no data'}\n"
                f"  imu2 (forearm): {'ok' if fresh[2] else 'no data'}")
            return []

        pose = tracker.solve(Rt, Ra, dist if use_dist[0] else None) if tracker.v is not None \
            else tracker.rest_pose()
        R = pose["Rdisp"]

        torso = [P["sh_a"], P["sh_b"], P["hip_b"], P["hip_a"], P["sh_a"],
                 P["neck"], P["pelvis"]]
        set3d(art["torso"], [R @ p for p in torso])
        set3d(art["neck"], [R @ P["neck"], R @ P["head"]])
        set3d(art["head"], [R @ P["head"]] * 2)
        set3d(art["other"], [R @ P["sh_b"], R @ other_hand])
        arm = [R @ pose["S"], R @ pose["elbow"], R @ pose["wrist"]]
        set3d(art["arm"], arm)
        set3d(art["hand"], [arm[2], R @ pose["hand"]])
        trail.append(arm[2])
        set3d(art["trail"], list(trail))

        cal = ("neutral " + ("OK" if tracker.v is not None else "-- press N") + "   forward "
               + ("OK" if tracker.psi is not None else "-- press F"))
        d_txt = f"{dist:.2f} m  (RSSI {rssi:.0f} dBm)" if dist is not None else "no link"
        lines = [
            f"Calibration : {cal}",
            f"Distance    : {d_txt}   [{'ON' if use_dist[0] else 'OFF'}]",
            f"Elbow solve : {pose['mode']}",
            f"Elbow bend  : {pose['flex']:5.1f} deg",
            f"Upper arm   : {pose['shoulder_ang']:5.1f} deg from hanging",
            "Keys: N neutral | F forward | D distance on/off | Q quit",
        ]
        art["info"].set_text("\n".join(lines))
        return []

    anim = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
    return fig, anim


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Torso + arm simulation from 2 IMUs")
    ap.add_argument("--side", choices=["right", "left"], default="right",
                    help="which arm imu2 is worn on")
    ap.add_argument("--mount", choices=list(MOUNTS), default="flat",
                    help="how the torso board (imu1) is oriented on the body")
    ap.add_argument("--elbow-mode", choices=["hinge", "cone"], default="hinge",
                    help="hinge: use forearm roll as well as distance; cone: distance only")
    ap.add_argument("--no-distance", action="store_true", help="start with distance off")
    ap.add_argument("--sim", action="store_true", help="synthetic data, no hardware")
    ap.add_argument("--sim-noise-db", type=float, default=0.5,
                    help="RSSI noise in the simulator; try 3 to see how a real link behaves")
    args = ap.parse_args()

    state = State()
    tracker = ArmTracker(args.side, args.mount, args.elbow_mode)
    stop = threading.Event()
    use_dist = [not args.no_distance]

    if args.sim:
        sim = Sim(args.side, args.mount, args.sim_noise_db)
        th = threading.Thread(target=sim_loop, args=(state, tracker, sim, stop), daemon=True)
        print("Simulation: auto-calibrates at 3 s (N) and 7 s (F), then moves freely.")
    else:
        th = threading.Thread(target=receiver_loop, args=(state, stop), daemon=True)
    th.start()

    fig, anim = run_viewer(state, tracker, args.side, use_dist)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        th.join(timeout=2)
        print("Packets received:", state.counts)


if __name__ == "__main__":
    main()
