"""Load a lidar session recorded by lidar_recorder.py (numpy only, no Pi dependencies).

    s = load_session("lidar_recordings/2026-10-07_153012_apartment1")
    for f in s.frames:           # f.t_mono, f.angles (rad), f.ranges (m, NaN = invalid), f.flags
        xy = f.xy()              # (M, 2) points, x forward, y left
"""
import glob
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Frame:
    frame_id: int
    t_mono: float
    t_wall: float
    angles: np.ndarray  # rad, CCW from forward (flip applied)
    ranges: np.ndarray  # m, NaN where the lidar reported no return
    intensity: np.ndarray
    flags: int
    crc_bad: int

    def xy(self, min_range=0.05, max_range=12.0, min_intensity=0) -> np.ndarray:
        r = self.ranges
        ok = np.isfinite(r) & (r >= min_range) & (r <= max_range) & (self.intensity >= min_intensity)
        return np.stack([r[ok] * np.cos(self.angles[ok]), r[ok] * np.sin(self.angles[ok])], axis=1).astype(np.float32)


@dataclass
class Session:
    path: str
    meta: dict
    frames: List[Frame]

    def timestamps(self) -> np.ndarray:
        return np.array([f.t_mono for f in self.frames])

    def summary(self) -> str:
        t = self.timestamps()
        dt = np.diff(t)
        pts = np.array([np.isfinite(f.ranges).sum() for f in self.frames])
        flagged = sum(1 for f in self.frames if f.flags)
        hz = 1 / np.median(dt) if len(dt) else float("nan")
        return (f"{len(self.frames)} frames, {t[-1] - t[0]:.1f} s, median {hz:.1f} Hz, "
                f"max gap {dt.max() if len(dt) else 0:.2f} s, median valid pts {np.median(pts):.0f}, "
                f"flagged frames {flagged}, crc_bad total {sum(f.crc_bad for f in self.frames)}")


def load_session(path: str, flip=None) -> Session:
    """flip=None uses the flag recorded in meta.json; pass True/False to override it."""
    meta_path = os.path.join(path, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if flip is None:
        flip = bool(meta.get("flip", False))
    chunks = sorted(p for p in glob.glob(os.path.join(path, "chunk_*.npz")))  # *.tmp never matches
    if not chunks:
        raise FileNotFoundError(f"no chunk_*.npz in {path}")
    frames = []
    for c in chunks:
        d = np.load(c)
        off = d["frame_offsets"]
        for i in range(len(d["frame_id"])):
            a, b = off[i], off[i + 1]
            ang = np.deg2rad(d["angle_cdeg"][a:b].astype(np.float32) / 100.0)
            if flip:
                ang = -ang
            rng = d["dist_mm"][a:b].astype(np.float32) / 1000.0
            rng[rng == 0] = np.nan
            frames.append(Frame(int(d["frame_id"][i]), float(d["t_mono"][i]), float(d["t_wall"][i]), ang, rng,
                                d["intensity"][a:b], int(d["flags"][i]), int(d["crc_bad"][i])))
    return Session(path, meta, frames)
