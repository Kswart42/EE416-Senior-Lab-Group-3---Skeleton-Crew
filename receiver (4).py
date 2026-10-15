"""
Receives IMU packets from the ESP32 boards over UDP and logs them to CSV.

Run:  python receiver.py
Stop: Ctrl+C

No extra libraries needed (standard library only).
"""
import csv
import socket
import time
from datetime import datetime

PORT = 4210
PRINT_EVERY = 10  # print every Nth packet per device (~10 Hz at 100 Hz send)

COLUMNS = ["pc_time", "id", "device_ms",
           "ax", "ay", "az", "gx", "gy", "gz",
           "roll", "pitch", "yaw"]

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(1.0)

filename = datetime.now().strftime("imu_log_%Y%m%d_%H%M%S.csv")
counts = {}
last_seen = {}

print(f"Listening on UDP port {PORT}. Logging to {filename}")
print("Waiting for data... (if nothing arrives, check the firewall note)\n")

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(COLUMNS)
    try:
        while True:
            try:
                data, addr = sock.recvfrom(512)
            except socket.timeout:
                continue

            pc_time = time.time()
            try:
                parts = data.decode().strip().split(",")
                if len(parts) != 11:
                    continue
                dev_id = parts[0]
                row = [f"{pc_time:.4f}"] + parts
            except UnicodeDecodeError:
                continue

            writer.writerow(row)

            counts[dev_id] = counts.get(dev_id, 0) + 1
            if dev_id not in last_seen:
                print(f"New device: id={dev_id} from {addr[0]}")
            last_seen[dev_id] = pc_time

            if counts[dev_id] % PRINT_EVERY == 0:
                ax, ay, az = (float(parts[i]) for i in (2, 3, 4))
                roll, pitch, yaw = (float(parts[i]) for i in (8, 9, 10))
                print(f"[imu{dev_id}] accel(m/s^2) x={ax:7.2f} y={ay:7.2f} z={az:7.2f} | "
                      f"roll={roll:7.2f} pitch={pitch:7.2f} yaw={yaw:7.2f} "
                      f"(pkts={counts[dev_id]})")
    except KeyboardInterrupt:
        print("\nStopped. Packets received:", counts)
