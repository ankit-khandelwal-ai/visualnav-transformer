#!/usr/bin/env python3
"""Live top-down view of a 2D lidar read directly over serial (no ROS).

Parses the 47-byte packet format used by the LDROBOT LD19 / LD06 family
(header 0x54, 0x2C; 12 points/packet; CRC8). The MentorPi's MS200 and LD19 both
run at 230400 baud on /dev/ldlidar. Packets failing CRC are dropped and counted,
so a high "bad" count means the wrong lidar/protocol or baud rate.

    pip install pyserial opencv-python numpy
    python3 visualize_lidar_live.py --port /dev/ldlidar --range 6
"""
import argparse
import time

import cv2
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
    ap.add_argument("--size", type=int, default=800, help="window size in px")
    ap.add_argument("--min-intensity", type=int, default=0)
    ap.add_argument("--flip", action="store_true", help="mirror angles (clockwise lidar)")
    args = ap.parse_args()

    lidar = LidarReader(args.port, args.baud)

    # Accumulate one revolution: points keyed by 1-degree bin, newest wins.
    pts = np.full((360, 2), np.nan)  # x forward, y left
    last_bin = None
    fps_t, revs = time.time(), 0
    rate = 0.0

    size = args.size
    scale = size / 2 / args.range  # px per meter
    cx = cy = size // 2
    base = np.zeros((size, size, 3), np.uint8)
    for r in range(1, int(args.range) + 1):
        cv2.circle(base, (cx, cy), int(r * scale), (60, 60, 60), 1)
        cv2.putText(base, f"{r}m", (cx + 3, cy - int(r * scale) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
    cv2.line(base, (cx, 0), (cx, size), (60, 60, 60), 1)
    cv2.line(base, (0, cy), (size, cy), (60, 60, 60), 1)
    win = "lidar (q to quit)"

    try:
        while True:
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
            img = base.copy()
            # x forward -> up, y left -> left
            px = (cx - pts[ok][:, 1] * scale).astype(int)
            py = (cy - pts[ok][:, 0] * scale).astype(int)
            for x, y in zip(px, py):
                if 0 <= x < size and 0 <= y < size:
                    cv2.circle(img, (x, y), 2, (255, 180, 0), -1)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img, f"{ok.sum()} pts  {rate:.1f} Hz  ok={lidar.good} bad={lidar.bad}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(win, img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        lidar.ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
