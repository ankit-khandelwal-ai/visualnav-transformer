"""ZeroMQ request/reply protocol between the robot (client) and the inference server.

The robot sends one preprocessed frame per step (REQ) and blocks until the server
replies with candidate waypoints (REP). The server binds, the robot connects out.

    robot.py  --RobotClient.request(FrameMsg)-->  server.py  (InferenceServer)
              <------------ ActionMsg -----------
"""
import json
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import zmq

DEFAULT_PORT = 5555
JPEG_QUALITY = 95


@dataclass
class FrameMsg:
    seq: int
    t_capture: float  # sender's clock; echoed back unchanged, never compared across machines
    image: np.ndarray  # RGB uint8, already cropped and resized for the model


@dataclass
class ActionMsg:
    seq: int  # echoes FrameMsg.seq
    t_capture: float  # echoes FrameMsg.t_capture
    waypoints: Optional[np.ndarray] = None  # (num_samples, T, 2) cumulative, in model units
    infer_ms: float = 0.0
    spread: float = 0.0  # std of the sampled trajectories (uncertainty signal)
    chosen: int = 0  # index, in the server's original sampling order, of the sample now at waypoints[0]
    status: str = "ok"  # "ok" | "warmup" | "error"
    error: str = ""
    velocity: Optional[list] = None  # [v m/s, w rad/s] direct command (joystick mode); used instead of waypoints
    closest_node: Optional[int] = None  # goal mode: topomap node the robot localized to this step
    reached_goal: Optional[bool] = None  # goal mode: localized to the goal node


def encode_frame(msg: FrameMsg) -> list:
    bgr = cv2.cvtColor(msg.image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    header = json.dumps({"seq": msg.seq, "t_capture": msg.t_capture}).encode()
    return [header, buf.tobytes()]


def decode_frame(parts: list) -> FrameMsg:
    header = json.loads(parts[0])
    bgr = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
    return FrameMsg(header["seq"], header["t_capture"], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def encode_action(msg: ActionMsg) -> bytes:
    return json.dumps({
        "seq": msg.seq,
        "t_capture": msg.t_capture,
        "waypoints": None if msg.waypoints is None else np.asarray(msg.waypoints).tolist(),
        "infer_ms": msg.infer_ms,
        "spread": msg.spread,
        "chosen": msg.chosen,
        "status": msg.status,
        "error": msg.error,
        "velocity": msg.velocity,
        "closest_node": msg.closest_node,
        "reached_goal": msg.reached_goal,
    }).encode()


def decode_action(data: bytes) -> ActionMsg:
    d = json.loads(data)
    wp = None if d["waypoints"] is None else np.array(d["waypoints"], dtype=np.float32)
    return ActionMsg(seq=d["seq"], t_capture=d["t_capture"], waypoints=wp, infer_ms=d["infer_ms"],
                     spread=d["spread"], chosen=d.get("chosen", 0), status=d["status"], error=d["error"],
                     velocity=d.get("velocity"), closest_node=d.get("closest_node"),
                     reached_goal=d.get("reached_goal"))


class RobotClient:
    """REQ socket. request() returns the server's ActionMsg, or None if it timed out."""

    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout_ms: int = 2000):
        self.addr = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.sock = None
        self._connect()

    def _connect(self):
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def request(self, frame: FrameMsg) -> Optional[ActionMsg]:
        self.sock.send_multipart(encode_frame(frame))
        if self.sock.poll(self.timeout_ms, zmq.POLLIN):
            return decode_action(self.sock.recv())
        self._connect()  # a REQ socket that missed its reply is stuck, so start a fresh one
        return None

    def close(self):
        self.sock.close(linger=0)


class InferenceServer:
    """REP socket. Every frame received must be answered with exactly one send_action()."""

    def __init__(self, port: int = DEFAULT_PORT):
        self.sock = zmq.Context.instance().socket(zmq.REP)
        self.sock.bind(f"tcp://*:{port}")

    def recv_frame(self, timeout_ms: Optional[int] = None) -> Optional[FrameMsg]:
        if timeout_ms is not None and not self.sock.poll(timeout_ms, zmq.POLLIN):
            return None
        return decode_frame(self.sock.recv_multipart())

    def send_action(self, action: ActionMsg):
        self.sock.send(encode_action(action))

    def close(self):
        self.sock.close(linger=0)
