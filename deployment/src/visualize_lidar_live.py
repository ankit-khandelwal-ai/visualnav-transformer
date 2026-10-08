#!/usr/bin/env python3
"""Live top-down view of a 2D lidar read directly over serial (no ROS).

Parses the 47-byte packet format used by the LDROBOT LD19 / LD06 family
(header 0x54, 0x2C; 12 points/packet; CRC8). The MentorPi's MS200 and LD19 both
run at 230400 baud on /dev/ldlidar. Packets failing CRC are dropped and counted,
so a high "bad" count means the wrong lidar/protocol or baud rate.

    pip install pyserial matplotlib numpy
    python3 visualize_lidar_live.py --port /dev/ldlidar --range 6
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import serial

HEADER = 0x54
VER_LEN = 0x2C
PKT_LEN = 47
POINTS_PER_PKT = 12


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def parse_packet(p: bytes):
    """Return (angles_rad, dists_m, intensities) for one valid packet."""
    start = int.from_bytes(p[4:6], "little") / 100.0
    end = int.from_bytes(p[42:44], "little") / 100.0
    if end < start:
        end += 360.0
    step = (end - start) / (POINTS_PER_PKT - 1)
    ang, dist, inten = [], [], []
    for i in range(POINTS_PER_PKT):
        o = 6 + 3 * i
        d = int.from_bytes(p[o:o + 2], "little") / 1000.0
        ang.append(np.deg2rad((start + step * i) % 360.0))
        dist.append(d)
        inten.append(p[o + 2])
    return np.array(ang), np.array(dist), np.array(inten)


class LidarReader:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.buf = bytearray()
        self.good = 0
        self.bad = 0

    def read_packets(self):
        """Yield every complete valid packet currently available."""
        self.buf += self.ser.read(max(self.ser.in_waiting, 1))
        while len(self.buf) >= PKT_LEN:
            if self.buf[0] != HEADER or self.buf[1] != VER_LEN:
                i = self.buf.find(bytes([HEADER, VER_LEN]), 1)
                del self.buf[: i if i != -1 else len(self.buf) - 1]
                continue
            pkt = bytes(self.buf[:PKT_LEN])
            if crc8(pkt[:-1]) != pkt[-1]:
                self.bad += 1
                del self.buf[:1]  # resync
                continue
            del self.buf[:PKT_LEN]
            self.good += 1
            yield pkt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ldlidar")
    ap.add_argument("--baud", type=int, default=230400)
    ap.add_argument("--range", type=float, default=6.0, help="plot half-extent (m)")
    ap.add_argument("--min-intensity", type=int, default=0)
    ap.add_argument("--flip", action="store_true", help="mirror angles (clockwise lidar)")
    args = ap.parse_args()

    lidar = LidarReader(args.port, args.baud)

    # Accumulate one revolution: points keyed by 1-degree bin, newest wins.
    pts = np.full((360, 2), np.nan)  # x forward, y left
    last_bin = None
    fps_t, revs = time.time(), 0
    rate = 0.0

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter([], [], s=4, c="tab:blue")
    ax.plot(0, 0, "r^", markersize=10)
    ax.set_xlim(args.range, -args.range)  # plot x = y_left (inverted so left is left)
    ax.set_ylim(-args.range, args.range)  # plot y = x_forward
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("y (m, left)")
    ax.set_ylabel("x (m, forward)")
    title = ax.set_title("waiting for data ...")
    plt.ion()
    plt.show()

    try:
        while plt.fignum_exists(fig.number):
            for pkt in lidar.read_packets():
                ang, dist, inten = parse_packet(pkt)
                if args.flip:
                    ang = -ang
                for a, d, it in zip(ang, dist, inten):
                    b = int(np.rad2deg(a) % 360)
                    if d > 0.0 and it >= args.min_intensity:
                        pts[b] = (d * np.cos(a), d * np.sin(a))
                    else:
                        pts[b] = np.nan
                    if last_bin is not None and b < last_bin:
                        revs += 1
                    last_bin = b
            now = time.time()
            if now - fps_t >= 1.0:
                rate, revs, fps_t = revs / (now - fps_t), 0, now
            ok = ~np.isnan(pts[:, 0])
            sc.set_offsets(pts[ok][:, ::-1])  # (y, x)
            title.set_text(
                f"{args.port}  {ok.sum()} pts  {rate:.1f} Hz  "
                f"pkts ok={lidar.good} bad={lidar.bad}"
            )
            fig.canvas.draw_idle()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        lidar.ser.close()


if __name__ == "__main__":
    main()
