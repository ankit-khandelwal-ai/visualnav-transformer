"""Background lidar recorder for robot.py's --lidar-collect mode (no ROS, serial only).

Reads the LD19-style packets via visualize_lidar_live.LidarReader, groups them into full
revolutions, and writes crash-safe chunked .npz files plus meta.json. See
../LIDAR_RECORDING_PLAN.md for the format. Raw units on disk; the Mac-side loader converts.

Session layout:
    <dir>/<YYYY-mm-dd_HHMMSS>[_label]/
        meta.json              written at start, rewritten (atomically) at the end
        chunk_0000.npz ...     ~chunk_seconds of frames each (flat arrays + frame_offsets)
        commands.jsonl         robot (v, w) commands with t_mono, for aligning motion to scans
"""
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

from visualize_lidar_live import PKT_LEN, POINTS_PER_PKT

SCHEMA_VERSION = 1
FLAG_LOW_POINTS, FLAG_ANGLE_GAP, FLAG_TIMING = 1, 2, 4
MAX_GAP_CDEG = 1000  # 10 deg between consecutive packet starts counts as a gap
MIN_FREE_BYTES = 200 * 1024 * 1024


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=here,
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return {"commit": rev, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


class LidarRecorder:
    def __init__(self, out_dir, port, baud=230400, label="", chunk_seconds=30.0, flip=False, args=None):
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")
        self.session = os.path.join(out_dir, stamp + (f"_{slug}" if slug else ""))
        if os.path.exists(self.session):
            raise RuntimeError(f"{self.session} already exists")
        os.makedirs(self.session)

        from visualize_lidar_live import LidarReader  # opens the port; fails early if it's wrong
        self.reader = LidarReader(port, baud)
        self.chunk_seconds = chunk_seconds
        self.stop_event = threading.Event()
        self.q = queue.Queue(maxsize=4)
        self.stop_reason = "stopped"
        self.frames = 0
        self.chunks = 0
        self.dropped_chunks = 0
        self.dropped_partial = 0
        self.last_error = None
        self.t_start_mono = time.monotonic()
        self.cmd_file = open(os.path.join(self.session, "commands.jsonl"), "w")
        self.meta = {
            "schema_version": SCHEMA_VERSION, "port": port, "baud": baud, "label": label,
            "lidar_model": "LD19-compatible, unverified",
            "packet": {"header": [0x54, 0x2C], "length": PKT_LEN, "points_per_packet": POINTS_PER_PKT,
                       "angle_unit": "centi-degrees", "dist_unit": "mm"},
            "flip": flip, "chunk_seconds": chunk_seconds, "args": args or {},
            "git": _git_info(), "hostname": platform.node(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "start_wall": datetime.now().isoformat(timespec="seconds"),
            "start_mono": self.t_start_mono,
        }
        import serial
        self.meta["pyserial"] = serial.__version__
        _atomic_json(os.path.join(self.session, "meta.json"), self.meta)
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()
        self.reader_thread.start()

    # --- called from robot.py's control loop ---------------------------------------------------
    def log_command(self, v, w):
        self.cmd_file.write(json.dumps({"t_mono": round(time.monotonic(), 4), "v": v, "w": w}) + "\n")

    def status(self) -> str:
        return (f"lidar: {self.frames} revs, {self.reader.good} pkts ok / {self.reader.bad} bad"
                + (f", ERROR {self.last_error}" if self.last_error else ""))

    # --- reader thread: packets -> revolutions -> chunks ---------------------------------------
    def _new_chunk(self):
        return {"frame_id": [], "t_mono": [], "t_wall": [], "n_packets": [], "crc_bad": [],
                "flags": [], "pkt_t": [], "pkt_counts": [], "angle": [], "dist": [], "inten": [],
                "pkt_idx": []}

    def _read_loop(self):
        chunk = self._new_chunk()
        chunk_t0 = time.monotonic()
        # current (in-progress) revolution
        cur = None
        prev_start = None
        bad_at_last_frame = 0
        frame_id = 0
        last_pkt_t = time.monotonic()
        try:
            while not self.stop_event.is_set():
                got = False
                for pkt in self.reader.read_packets():
                    got = True
                    t_m, t_w = time.monotonic(), time.time()
                    last_pkt_t = t_m
                    start = int.from_bytes(pkt[4:6], "little")  # centi-degrees
                    end = int.from_bytes(pkt[42:44], "little")
                    if prev_start is not None and start < prev_start - 18000:  # wrapped past 360
                        if cur is not None:  # the revolution that just ended
                            self._finish_frame(chunk, cur, frame_id, self.reader.bad - bad_at_last_frame)
                            frame_id += 1
                            bad_at_last_frame = self.reader.bad
                        cur = {"t_mono": t_m, "t_wall": t_w, "pkt_t": [], "pkt_starts": [],
                               "angle": [], "dist": [], "inten": [], "pkt_idx": []}
                    prev_start = start
                    if cur is None:
                        continue  # still in the partial first revolution: discard
                    span = end + 36000 if end < start else end
                    step = (span - start) / (POINTS_PER_PKT - 1)
                    n = len(cur["pkt_t"])
                    cur["pkt_t"].append(t_m)
                    cur["pkt_starts"].append(start)
                    for i in range(POINTS_PER_PKT):
                        o = 6 + 3 * i
                        cur["angle"].append(int(round(start + step * i)) % 36000)
                        cur["dist"].append(int.from_bytes(pkt[o:o + 2], "little"))
                        cur["inten"].append(pkt[o + 2])
                        cur["pkt_idx"].append(n)
                if chunk["frame_id"] and time.monotonic() - chunk_t0 >= self.chunk_seconds:
                    self._submit(chunk)
                    chunk, chunk_t0 = self._new_chunk(), time.monotonic()
                    if self._disk_low():
                        self.stop_reason = "disk low"
                        break
                if not got and time.monotonic() - last_pkt_t > 5.0:
                    self.last_error = "no valid packets for 5 s"
                    self.stop_reason = self.last_error
                    break
        except Exception as e:  # serial unplugged etc.: keep what we have
            self.last_error = repr(e)
            self.stop_reason = f"error: {e!r}"
        if cur is not None and cur["pkt_t"]:
            self.dropped_partial += 1
        if chunk["frame_id"]:
            self._submit(chunk)
        self.q.put(None)  # tell the writer to finish

    def _finish_frame(self, chunk, cur, frame_id, crc_bad):
        starts = np.array(cur["pkt_starts"])
        gaps = np.diff(starts)
        flags = 0
        if len(cur["dist"]) < 0.5 * 360:  # far fewer points than a normal revolution
            flags |= FLAG_LOW_POINTS
        if len(gaps) and gaps.max() > MAX_GAP_CDEG:
            flags |= FLAG_ANGLE_GAP
        ts = np.diff(cur["pkt_t"])
        if len(ts) and ts.max() > 0.05:
            flags |= FLAG_TIMING
        chunk["frame_id"].append(frame_id)
        chunk["t_mono"].append(cur["t_mono"])
        chunk["t_wall"].append(cur["t_wall"])
        chunk["n_packets"].append(len(cur["pkt_t"]))
        chunk["crc_bad"].append(min(crc_bad, 65535))
        chunk["flags"].append(flags)
        chunk["pkt_t"].append(np.array(cur["pkt_t"], np.float64))
        chunk["pkt_counts"].append(len(cur["pkt_t"]))
        chunk["angle"].append(np.array(cur["angle"], np.uint16))
        chunk["dist"].append(np.array(cur["dist"], np.uint16))
        chunk["inten"].append(np.array(cur["inten"], np.uint8))
        chunk["pkt_idx"].append(np.array(cur["pkt_idx"], np.uint16))
        self.frames += 1

    def _submit(self, chunk):
        try:
            self.q.put_nowait(chunk)
        except queue.Full:  # never stall the serial read on disk
            self.dropped_chunks += 1

    def _disk_low(self):
        st = os.statvfs(self.session)
        return st.f_bavail * st.f_frsize < MIN_FREE_BYTES

    # --- writer thread --------------------------------------------------------------------------
    def _write_loop(self):
        while True:
            chunk = self.q.get()
            if chunk is None:
                return
            n_pts = [len(a) for a in chunk["angle"]]
            data = dict(
                frame_id=np.array(chunk["frame_id"], np.uint32),
                t_mono=np.array(chunk["t_mono"], np.float64),
                t_wall=np.array(chunk["t_wall"], np.float64),
                n_packets=np.array(chunk["n_packets"], np.uint16),
                crc_bad=np.array(chunk["crc_bad"], np.uint16),
                flags=np.array(chunk["flags"], np.uint8),
                frame_offsets=np.concatenate([[0], np.cumsum(n_pts)]).astype(np.int64),
                pkt_offsets=np.concatenate([[0], np.cumsum(chunk["pkt_counts"])]).astype(np.int64),
                pkt_t_mono=np.concatenate(chunk["pkt_t"]),
                angle_cdeg=np.concatenate(chunk["angle"]),
                dist_mm=np.concatenate(chunk["dist"]),
                intensity=np.concatenate(chunk["inten"]),
                pkt_idx=np.concatenate(chunk["pkt_idx"]),
            )
            final = os.path.join(self.session, f"chunk_{self.chunks:04d}.npz")
            tmp = final + ".tmp"
            with open(tmp, "wb") as f:  # uncompressed: saves CPU on the Pi
                np.savez(f, **data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, final)
            self.chunks += 1

    # --- shutdown -------------------------------------------------------------------------------
    def close(self):
        self.stop_event.set()
        self.reader_thread.join(timeout=5)
        self.writer_thread.join(timeout=30)
        self.reader.ser.close()
        self.cmd_file.close()
        self.meta.update({
            "end_wall": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.monotonic() - self.t_start_mono, 2),
            "frames": self.frames, "chunks": self.chunks,
            "packets_ok": self.reader.good, "packets_bad": self.reader.bad,
            "dropped_chunks": self.dropped_chunks, "dropped_partial_frames": self.dropped_partial,
            "stop_reason": self.stop_reason,
        })
        _atomic_json(os.path.join(self.session, "meta.json"), self.meta)
        return self.meta
