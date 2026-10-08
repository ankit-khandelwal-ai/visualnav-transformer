"""Latest lidar revolution, read straight from serial in a background thread (for robot.py --lidar-stream).

Same LD19-style packets as lidar_recorder.py. Only one process can own the serial port, so
--lidar-stream and --lidar-collect are mutually exclusive.
"""
import threading
import time
from typing import Optional

import numpy as np

from visualize_lidar_live import POINTS_PER_PKT, LidarReader


class LidarScanner:
    def __init__(self, port: str, baud: int = 230400, flip: bool = False):
        self.reader = LidarReader(port, baud)
        self.flip = flip
        self._lock = threading.Lock()
        self._scan: Optional[np.ndarray] = None
        self._t = 0.0
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        ang, rng = [], []
        prev_start = None
        while not self._stop:
            for pkt in self.reader.read_packets():
                start = int.from_bytes(pkt[4:6], "little")
                end = int.from_bytes(pkt[42:44], "little")
                if prev_start is not None and start < prev_start - 18000 and ang:  # wrapped: revolution done
                    a = np.deg2rad(np.array(ang, np.float32) / 100.0)
                    r = np.array(rng, np.float32) / 1000.0
                    r[r == 0] = np.nan
                    if self.flip:
                        a = -a
                    with self._lock:
                        self._scan, self._t = np.stack([a, r], axis=1), time.monotonic()
                    ang, rng = [], []
                prev_start = start
                span = end + 36000 if end < start else end
                step = (span - start) / (POINTS_PER_PKT - 1)
                for i in range(POINTS_PER_PKT):
                    ang.append(int(round(start + step * i)) % 36000)
                    rng.append(int.from_bytes(pkt[6 + 3 * i:8 + 3 * i], "little"))

    def latest(self):
        """(scan (N,2) [angle rad, range m] or None, age in seconds)."""
        with self._lock:
            return (None, float("inf")) if self._scan is None else (self._scan.copy(), time.monotonic() - self._t)

    def status(self) -> str:
        return f"lidar: {self.reader.good} pkts ok / {self.reader.bad} bad"

    def close(self):
        self._stop = True
        time.sleep(0.1)
        self.reader.ser.close()
