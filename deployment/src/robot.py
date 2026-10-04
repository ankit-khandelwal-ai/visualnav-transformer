"""Robot-side loop: camera frame -> crop for NoMaD -> server -> waypoints -> wheel commands.

On the Pi:
    python robot.py --host <mac-ip>               # dry run: prints v/w, motors untouched
    python robot.py --host <mac-ip> --arm         # drives the wheels (wheels off the ground first!)
On the Mac (loopback test with the webcam):
    python robot.py --host localhost
"""
import argparse
import glob
import json
import math
import os
import sys
import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

from protocol import DEFAULT_PORT, ActionMsg, FrameMsg, RobotClient

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "../config/robot.yaml")) as f:
    _robot_cfg = yaml.safe_load(f)

MAX_V = _robot_cfg["max_v"]  # m/s
MAX_W = _robot_cfg["max_w"]  # rad/s
RATE = _robot_cfg["frame_rate"]  # Hz; NoMaD's context frames are spaced 1/RATE apart
DT = 1.0 / RATE
EPS = 1e-8

IMAGE_SIZE = (96, 96)  # nomad.yaml image_size (width, height)
IMAGE_ASPECT_RATIO = 4 / 3  # training images are center-cropped to 4:3
WAYPOINT_IDX = 2  # which of the T predicted waypoints to steer toward (repo default)

# ---- TODO(calibration): fill these in from motor_test.py and measurements -------------------
WHEEL_RADIUS_M = 0.0325  # PLACEHOLDER
TRACK_WIDTH_M = 0.30  # PLACEHOLDER: distance between left and right wheels
MAX_WHEEL_RPS = 1.0
# motor id -> (side, sign). sign flips motors so that +rps drives the robot forward.
# Layout: 1 = front-left, 2 = back-left, 3 = front-right, 4 = back-right. Signs verified with wheels up.
WHEEL_MAP = {1: ("left", -1), 2: ("left", -1), 3: ("right", +1), 4: ("right", +1)}
CALIBRATED = False
# ----------------------------------------------------------------------------------------------


def preprocess_frame(bgr: np.ndarray) -> np.ndarray:
    """Same preprocessing as training: center-crop to 4:3, resize to the model input size. Returns RGB."""
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    w, h = img.size
    if w > h:
        crop_w, crop_h = min(w, int(h * IMAGE_ASPECT_RATIO)), h
    else:
        crop_w, crop_h = w, min(h, int(w / IMAGE_ASPECT_RATIO))
    left, top = (w - crop_w) // 2, (h - crop_h) // 2
    img = img.crop((left, top, left + crop_w, top + crop_h)).resize(IMAGE_SIZE)
    return np.asarray(img)


class Camera:
    """Reads the camera in a background thread so latest() never returns a stale buffered frame."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock = threading.Lock()
        self._frame, self._t = None, 0.0
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()
        deadline = time.monotonic() + 5
        while self._frame is None:
            if time.monotonic() > deadline:
                raise RuntimeError("Camera produced no frames")
            time.sleep(0.01)

    def _run(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame, self._t = frame, time.monotonic()

    def latest(self) -> Tuple[np.ndarray, float]:
        with self._lock:
            return self._frame.copy(), self._t

    def close(self):
        self._stop = True
        time.sleep(0.1)
        self.cap.release()


class RobotControl:
    """Turns server actions into wheel commands. Owns all kinematics and speed limits."""

    def __init__(self, board=None):
        self.board = board  # None = dry run
        self.safety_filter: Callable[[float, float], Tuple[float, float]] = lambda v, w: (v, w)

    # --- action -> velocity -------------------------------------------------------------------
    @staticmethod
    def waypoint_to_vel(waypoint: np.ndarray) -> Tuple[float, float]:
        """PD controller from deployment/src/pd_controller.py. waypoint = (dx, dy) in model units."""
        dx, dy = (float(waypoint[0]) * MAX_V / RATE, float(waypoint[1]) * MAX_V / RATE)  # model units -> m
        if abs(dx) < EPS and abs(dy) < EPS:
            v, w = 0.0, 0.0
        elif abs(dx) < EPS:
            v, w = 0.0, math.copysign(math.pi / (2 * DT), dy)
        else:
            v, w = dx / DT, math.atan(dy / dx) / DT
        return float(np.clip(v, 0, MAX_V)), float(np.clip(w, -MAX_W, MAX_W))

    def action_to_vel(self, action: ActionMsg, sample: int = 0, waypoint_idx: int = WAYPOINT_IDX) -> Tuple[float, float]:
        return self.waypoint_to_vel(action.waypoints[sample][waypoint_idx])

    # --- velocity -> wheels -------------------------------------------------------------------
    @staticmethod
    def vel_to_wheel_rps(v: float, w: float) -> dict:
        """Forward + yaw only. Differential-drive approximation of the chassis."""
        side_speed = {"left": v - w * TRACK_WIDTH_M / 2, "right": v + w * TRACK_WIDTH_M / 2}
        rps = {}
        for motor, (side, sign) in WHEEL_MAP.items():
            r = sign * side_speed[side] / (2 * math.pi * WHEEL_RADIUS_M)
            rps[motor] = float(np.clip(r, -MAX_WHEEL_RPS, MAX_WHEEL_RPS))
        return rps

    def set_wheel_rps(self, rps: dict):
        """Raw per-motor command {motor_id: rps}, bypassing WHEEL_MAP signs. No-op on a dry run."""
        if self.board is not None:
            self.board.set_motor_speed([[m, r] for m, r in rps.items()])

    def set_velocity(self, v: float, w: float) -> Tuple[float, float]:
        v, w = self.safety_filter(v, w)
        self.set_wheel_rps(self.vel_to_wheel_rps(v, w))
        return v, w

    def act(self, action: ActionMsg) -> Tuple[float, float]:
        return self.set_velocity(*self.action_to_vel(action))

    def command_velocity(self, v: float, w: float) -> Tuple[float, float]:
        """Direct (v, w) command, e.g. from joystick mode. The robot clamps it to its own limits."""
        return self.set_velocity(float(np.clip(v, -MAX_V, MAX_V)), float(np.clip(w, -MAX_W, MAX_W)))

    def stop(self):
        for _ in range(3):  # repeat in case a packet is dropped
            self.set_wheel_rps({m: 0.0 for m in WHEEL_MAP})
            time.sleep(0.02)


def find_port() -> Optional[str]:
    if os.path.exists("/dev/rrc"):
        return "/dev/rrc"
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return candidates[0] if candidates else None


def open_board(port: Optional[str]):
    from ros_robot_controller_sdk import Board

    port = port or find_port()
    if port is None:
        sys.exit("No serial port found for the RRC Lite board")
    return Board(device=port)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True, help="IP of the machine running server.py")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--serial-port", default=None, help="RRC Lite serial device (default: auto)")
    p.add_argument("--arm", action="store_true", help="actually drive the wheels (default: dry run)")
    p.add_argument("--timeout-ms", type=int, default=2000, help="give up on a server reply after this long")
    p.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = run until Ctrl-C)")
    p.add_argument("--log", default=None, help="write one JSON line per step to this file")
    args = p.parse_args()

    if args.arm and not CALIBRATED:
        print("WARNING: WHEEL_MAP / wheel geometry are placeholders. Check wheel directions with wheels off the ground.")
    robot = RobotControl(open_board(args.serial_port) if args.arm else None)
    camera = Camera(args.camera)
    client = RobotClient(args.host, args.port, args.timeout_ms)
    log = open(args.log, "w") if args.log else None
    print(f"{'ARMED' if args.arm else 'DRY RUN'}: sending frames to {args.host}:{args.port} at {RATE} Hz. Ctrl-C to stop.")

    seq = 0
    try:
        while args.max_steps == 0 or seq < args.max_steps:
            t_step = time.monotonic()
            bgr, t_capture = camera.latest()
            frame = FrameMsg(seq=seq, t_capture=t_capture, image=preprocess_frame(bgr))

            t_send = time.monotonic()
            action = client.request(frame)
            rtt_ms = (time.monotonic() - t_send) * 1000

            if action is None:
                robot.stop()
                v = w = 0.0
                status = "timeout"
            elif action.status == "ok" and action.velocity is not None:
                v, w = robot.command_velocity(*action.velocity)
                status = "joystick"
            elif action.status != "ok" or action.waypoints is None:
                robot.stop()
                v = w = 0.0
                status = action.status if not action.error else f"{action.status}: {action.error}"
            else:
                v, w = robot.act(action)
                status = "ok"

            rec = {"seq": seq, "rtt_ms": round(rtt_ms, 1), "infer_ms": None if action is None else round(action.infer_ms, 1),
                   "spread": None if action is None else round(action.spread, 4), "v": round(v, 3), "w": round(w, 3), "status": status}
            print(rec)
            if log:
                log.write(json.dumps(rec) + "\n")
                log.flush()

            seq += 1
            time.sleep(max(0.0, DT - (time.monotonic() - t_step)))
    except KeyboardInterrupt:
        pass
    finally:
        robot.stop()
        camera.close()
        client.close()
        if log:
            log.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
