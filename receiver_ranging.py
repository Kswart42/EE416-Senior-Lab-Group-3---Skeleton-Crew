"""
Receives IMU + ranging packets from the ESP32 boards over UDP, logs them to
CSV, and draws live graphs in two windows:
  1) "Live IMU data"  - accel / gyro / angles per board
  2) "Ranging"        - signal strength (RSSI) and estimated distance

Run:   python receiver_ranging.py
Stop:  close a plot window (or Ctrl+C in the terminal)

Needs matplotlib:   python -m pip install matplotlib

Packet format (12 fields): id,device_ms,ax,ay,az,gx,gy,gz,roll,pitch,yaw,rssi
(11-field packets from older firmware are still accepted; they just have
no ranging data.)

------------------------------------------------------------------------
DISTANCE CALIBRATION  (read this, it matters)
------------------------------------------------------------------------
Distance is estimated from signal strength with the log-distance model:

    distance = 10 ** ((RSSI_AT_1M - rssi) / (10 * PATH_LOSS_EXPONENT))

Both constants depend on your boards, antennas, enclosure and room, so
calibrate them:
  1) Put the boards exactly 1 m apart, line of sight, and let the RSSI
     settle. Set RSSI_AT_1M to the average RSSI shown in the terminal.
  2) Put them a known farther distance apart (e.g. 4 m). Then
         PATH_LOSS_EXPONENT = (RSSI_AT_1M - rssi_at_d) / (10 * log10(d))
     Typical values: ~2 in open space, 2.5 to 4 indoors.
Expect rough estimates (errors of a few metres indoors) because people,
walls and orientation all change the signal strength.
"""
import csv
import math
import socket
import threading
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PORT = 4210
NUM_DEVICES = 2        # IMU columns shown in the plot (ids 1..NUM_DEVICES)
WINDOW_SEC = 10        # seconds of history shown in the IMU plot
RANGE_WINDOW_SEC = 60  # seconds of history shown in the ranging plot
MAX_POINTS = 5000
PLOT_INTERVAL_MS = 50

# ---- Ranging calibration (see note above) ----
RSSI_AT_1M = -45.0       # dBm measured with the boards 1 m apart
PATH_LOSS_EXPONENT = 2.5 # 2 = free space, 2.5-4 = indoors

COLUMNS = ["pc_time", "id", "device_ms",
           "ax", "ay", "az", "gx", "gy", "gz",
           "roll", "pitch", "yaw", "rssi", "dist_m"]

KEYS = ("ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw")

lock = threading.Lock()
buffers = {
    i: {k: deque(maxlen=MAX_POINTS) for k in ("t",) + KEYS}
    for i in range(1, NUM_DEVICES + 1)
}
# Ranging data per device (only samples that actually have a link)
ranging = {
    i: {k: deque(maxlen=MAX_POINTS) for k in ("t", "rssi", "dist")}
    for i in range(1, NUM_DEVICES + 1)
}
counts = {}
stop_event = threading.Event()


def rssi_to_distance(rssi):
    return 10 ** ((RSSI_AT_1M - rssi) / (10.0 * PATH_LOSS_EXPONENT))


def receiver_loop(filename):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(1.0)
    print(f"Listening on UDP port {PORT}. Logging to {filename}")
    print("Waiting for data... (if nothing arrives, check the firewall note)\n")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        last_flush = time.time()

        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(512)
            except socket.timeout:
                continue

            pc_time = time.time()
            try:
                parts = data.decode().strip().split(",")
                if len(parts) not in (11, 12):
                    continue
                dev_id = int(parts[0])
                vals = [float(x) for x in parts[2:11]]
                rssi = float(parts[11]) if len(parts) == 12 else 0.0
            except (UnicodeDecodeError, ValueError):
                continue

            dist = rssi_to_distance(rssi) if rssi != 0.0 else None

            writer.writerow([f"{pc_time:.4f}"] + parts[:11] +
                            [f"{rssi:.1f}", "" if dist is None else f"{dist:.2f}"])
            if pc_time - last_flush > 1.0:
                f.flush()
                last_flush = pc_time

            if dev_id not in counts:
                print(f"New device: id={dev_id} from {addr[0]}")
            counts[dev_id] = counts.get(dev_id, 0) + 1

            if dist is not None and counts[dev_id] % 100 == 0:
                print(f"[imu{dev_id}] link RSSI {rssi:6.1f} dBm  ->  "
                      f"~{dist:5.2f} m (uncalibrated estimate)")

            buf = buffers.get(dev_id)
            if buf is None:
                continue
            with lock:
                buf["t"].append(pc_time)
                for key, v in zip(KEYS, vals):
                    buf[key].append(v)
                if dist is not None:
                    r = ranging[dev_id]
                    r["t"].append(pc_time)
                    r["rssi"].append(rssi)
                    r["dist"].append(dist)

    sock.close()


# --------------------------------------------------------------------------
# Plot 1: IMU data
# --------------------------------------------------------------------------
def build_imu_plot():
    fig, axes = plt.subplots(3, NUM_DEVICES, figsize=(6 * NUM_DEVICES, 8),
                             squeeze=False, sharex=True)
    fig.suptitle("Live IMU data")
    fig.canvas.manager.set_window_title("Live IMU data")

    groups = [
        ("Accel (m/s²)", ("ax", "ay", "az"), ("x", "y", "z")),
        ("Gyro (rad/s)", ("gx", "gy", "gz"), ("x", "y", "z")),
        ("Angles (°)",   ("roll", "pitch", "yaw"), ("roll", "pitch", "yaw")),
    ]

    lines = {}
    for col in range(NUM_DEVICES):
        dev = col + 1
        for row, (ylabel, keys, labels) in enumerate(groups):
            ax = axes[row][col]
            for key, label in zip(keys, labels):
                (line,) = ax.plot([], [], label=label, linewidth=1)
                lines[(dev, key)] = line
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlim(-WINDOW_SEC, 0)
            if row == 0:
                ax.set_title(f"imu{dev}")
            if row == 2:
                ax.set_xlabel("seconds ago")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes, lines, groups


def make_imu_updater(axes, lines, groups):
    def update(_frame):
        now = time.time()
        with lock:
            snap = {dev: {k: list(v) for k, v in b.items()}
                    for dev, b in buffers.items()}

        for col in range(NUM_DEVICES):
            dev = col + 1
            b = snap[dev]
            t = [x - now for x in b["t"]]
            start = 0
            while start < len(t) and t[start] < -WINDOW_SEC:
                start += 1
            t = t[start:]

            for row, (_, keys, _) in enumerate(groups):
                ax = axes[row][col]
                ymin, ymax = None, None
                for key in keys:
                    y = b[key][start:]
                    lines[(dev, key)].set_data(t, y)
                    if y:
                        lo, hi = min(y), max(y)
                        ymin = lo if ymin is None else min(ymin, lo)
                        ymax = hi if ymax is None else max(ymax, hi)
                if ymin is not None:
                    pad = max((ymax - ymin) * 0.1, 0.1)
                    ax.set_ylim(ymin - pad, ymax + pad)
        return []
    return update


# --------------------------------------------------------------------------
# Plot 2: ranging
# --------------------------------------------------------------------------
def build_ranging_plot():
    fig, (ax_rssi, ax_dist) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.canvas.manager.set_window_title("Ranging")
    fig.suptitle("Distance between boards (RSSI estimate)")

    ax_rssi.set_ylabel("RSSI (dBm)")
    ax_dist.set_ylabel("Estimated distance (m)")
    ax_dist.set_xlabel("seconds ago")
    for ax in (ax_rssi, ax_dist):
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-RANGE_WINDOW_SEC, 0)

    rssi_lines, dist_lines = {}, {}
    for dev in range(1, NUM_DEVICES + 1):
        (rl,) = ax_rssi.plot([], [], label=f"imu{dev}", linewidth=1)
        (dl,) = ax_dist.plot([], [], label=f"imu{dev}", linewidth=1)
        rssi_lines[dev], dist_lines[dev] = rl, dl
    ax_rssi.legend(loc="upper right", fontsize=8)
    ax_dist.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax_rssi, ax_dist, rssi_lines, dist_lines


def make_ranging_updater(fig, ax_rssi, ax_dist, rssi_lines, dist_lines):
    def update(_frame):
        now = time.time()
        with lock:
            snap = {dev: {k: list(v) for k, v in r.items()}
                    for dev, r in ranging.items()}

        all_rssi, all_dist, latest = [], [], []
        for dev, r in snap.items():
            t = [x - now for x in r["t"]]
            start = 0
            while start < len(t) and t[start] < -RANGE_WINDOW_SEC:
                start += 1
            t = t[start:]
            rs, ds = r["rssi"][start:], r["dist"][start:]
            rssi_lines[dev].set_data(t, rs)
            dist_lines[dev].set_data(t, ds)
            all_rssi += rs
            all_dist += ds
            if ds:
                latest.append(f"imu{dev}: {ds[-1]:.2f} m ({rs[-1]:.0f} dBm)")

        for ax, vals in ((ax_rssi, all_rssi), (ax_dist, all_dist)):
            if vals:
                lo, hi = min(vals), max(vals)
                pad = max((hi - lo) * 0.1, 1.0)
                ax.set_ylim(lo - pad, hi + pad)

        if latest:
            fig.suptitle("Distance between boards (RSSI estimate)   " + "   ".join(latest))
        else:
            fig.suptitle("Distance between boards: waiting for a link "
                         "(is imu2 powered and joined?)")
        return []
    return update


def main():
    filename = datetime.now().strftime("imu_log_%Y%m%d_%H%M%S.csv")
    thread = threading.Thread(target=receiver_loop, args=(filename,), daemon=True)
    thread.start()

    fig1, axes, lines, groups = build_imu_plot()
    fig2, ax_rssi, ax_dist, rssi_lines, dist_lines = build_ranging_plot()

    anim1 = FuncAnimation(fig1, make_imu_updater(axes, lines, groups),
                          interval=PLOT_INTERVAL_MS, blit=False,
                          cache_frame_data=False)
    anim2 = FuncAnimation(fig2, make_ranging_updater(fig2, ax_rssi, ax_dist,
                                                     rssi_lines, dist_lines),
                          interval=200, blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        thread.join(timeout=2)
        print("\nStopped. Packets received:", counts)
        print(f"Log saved to {filename}")


if __name__ == "__main__":
    main()
