"""Inference server: receives frames from robot.py, runs NoMaD (exploration mode), returns waypoints.

On the Mac (needs PYTHONPATH to include the diffusion_policy repo, see latency_bench.py):
    python server.py                  # picks mps/cuda/cpu automatically
    python server.py --num-samples 8 --device mps
    python server.py --joystick       # no NoMaD: drive the robot with the arrow keys (click the window first)
"""
import argparse
import collections
import os
import time

import cv2
import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image

from latency_bench import build_model, sync, to_tensor
from protocol import DEFAULT_PORT, ActionMsg, FrameMsg, InferenceServer

HERE = os.path.dirname(os.path.abspath(__file__))
NOMAD_CFG = os.path.join(HERE, "../../train/config/nomad.yaml")
DATA_CFG = os.path.join(HERE, "../../train/vint_train/data/data_config.yaml")
WEIGHTS = os.path.join(HERE, "../model_weights/nomad.pth")
with open(os.path.join(HERE, "../config/robot.yaml")) as _f:
    _robot_cfg = yaml.safe_load(_f)
MAX_V, MAX_W = _robot_cfg["max_v"], _robot_cfg["max_w"]  # the robot clamps to these too


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class NomadExplorer:
    """Goal-masked NoMaD: keeps a context of recent frames and samples collision-free trajectories."""

    def __init__(self, device: str, num_samples: int):
        self.device = torch.device(device)
        self.num_samples = num_samples
        self.cfg = yaml.safe_load(open(NOMAD_CFG))
        stats = yaml.safe_load(open(DATA_CFG))["action_stats"]
        self.stat_min, self.stat_max = np.array(stats["min"]), np.array(stats["max"])
        self.image_size = tuple(self.cfg["image_size"])
        self.num_iters = self.cfg["num_diffusion_iters"]
        self.model = build_model(self.cfg, WEIGHTS, self.device)
        self.sched = DDPMScheduler(num_train_timesteps=self.num_iters, beta_schedule="squaredcos_cap_v2",
                                   clip_sample=True, prediction_type="epsilon")
        self.context = collections.deque(maxlen=self.cfg["context_size"] + 1)

    @torch.no_grad()
    def step(self, frame: FrameMsg) -> ActionMsg:
        self.context.append(Image.fromarray(frame.image))
        if len(self.context) < self.context.maxlen:
            return ActionMsg(frame.seq, frame.t_capture, status="warmup")

        t0 = time.perf_counter()
        obs = to_tensor(list(self.context), self.image_size).to(self.device)
        goal = torch.randn((1, 3, *self.image_size), device=self.device)  # ignored: mask below
        mask = torch.ones(1, dtype=torch.long, device=self.device)
        cond = self.model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=mask)
        cond = cond.repeat(self.num_samples, 1) if cond.ndim == 2 else cond.repeat(self.num_samples, 1, 1)

        act = torch.randn((self.num_samples, self.cfg["len_traj_pred"], 2), device=self.device)
        self.sched.set_timesteps(self.num_iters)
        for k in self.sched.timesteps:
            noise = self.model("noise_pred_net", sample=act, timestep=k, global_cond=cond)
            act = self.sched.step(model_output=noise, timestep=k, sample=act).prev_sample
        sync(self.device)

        # same as train_utils.get_action: unnormalize deltas, then cumulative sum -> waypoints
        deltas = (act.cpu().numpy() + 1) / 2 * (self.stat_max - self.stat_min) + self.stat_min
        waypoints = np.cumsum(deltas, axis=1)
        return ActionMsg(frame.seq, frame.t_capture, waypoints=waypoints.astype(np.float32),
                         infer_ms=(time.perf_counter() - t0) * 1000, spread=float(waypoints.std(axis=0).mean()))


VIEW_SIZE = 480
WINDOW = "robot view (q / Esc to quit)"

# cv2.waitKeyEx codes for the arrow keys (macOS first, then Linux/GTK)
ARROWS = {63232: "up", 63233: "down", 63234: "left", 63235: "right",
          65362: "up", 65364: "down", 65361: "left", 65363: "right"}


class KeyPoller:
    """Reads keys from the OpenCV window. There are no key-up events, so an arrow key counts as held
    for `hold_s` after its last press or auto-repeat; only one arrow key can be active at a time."""

    def __init__(self, hold_s: float):
        self.hold_s = hold_s
        self.key, self.t = None, 0.0
        self.quit = False

    def poll(self):
        for _ in range(32):  # drain everything queued since the last call
            k = cv2.waitKeyEx(1)
            if k == -1:
                break
            if k in (27, ord("q")):
                self.quit = True
            elif k in (32, ord("s")):  # space or s: stop immediately
                self.key = None
            elif k in ARROWS:
                self.key, self.t = ARROWS[k], time.monotonic()

    def command(self, speed: float) -> list:
        """[v, w] for the held key, scaled by `speed` (fraction of the robot's max speeds)."""
        if self.key is None or time.monotonic() - self.t > self.hold_s:
            return [0.0, 0.0]
        v, w = {"up": (MAX_V, 0.0), "down": (-MAX_V, 0.0), "left": (0.0, MAX_W), "right": (0.0, -MAX_W)}[self.key]
        return [v * speed, w * speed]


def render(frame: FrameMsg, action: ActionMsg) -> np.ndarray:
    """Left: the frame the model saw, upscaled. Right: top-down view of the sampled trajectories
    (robot at the bottom, forward is up, left is left; sample 0, the one the robot follows, is green)."""
    img = cv2.resize(cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR), (VIEW_SIZE, VIEW_SIZE), interpolation=cv2.INTER_CUBIC)
    panel = np.full((VIEW_SIZE, VIEW_SIZE, 3), 30, np.uint8)
    origin = (VIEW_SIZE // 2, VIEW_SIZE - 20)
    cv2.circle(panel, origin, 5, (255, 255, 255), -1)

    if action.waypoints is not None:
        wp = action.waypoints  # (S, T, 2): x forward, y left
        scale = 0.9 * (VIEW_SIZE - 40) / max(1.0, float(np.abs(wp).max()))
        for s in range(wp.shape[0] - 1, -1, -1):  # draw sample 0 last so it's on top
            pts = [origin] + [(int(origin[0] - y * scale), int(origin[1] - x * scale)) for x, y in wp[s]]
            color = (0, 255, 0) if s == 0 else (140, 140, 140)
            cv2.polylines(panel, [np.array(pts, np.int32)], False, color, 2 if s == 0 else 1)
        cv2.circle(panel, pts[3], 6, (0, 255, 255), -1)  # sample 0, waypoint index 2: where the robot steers

    if action.velocity is not None:
        lines = [f"seq {frame.seq}  JOYSTICK", f"v {action.velocity[0]:+.2f} m/s  w {action.velocity[1]:+.2f} rad/s"]
    else:
        lines = [f"seq {frame.seq}  {action.status}", f"infer {action.infer_ms:.0f} ms  spread {action.spread:.2f}"]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return np.hstack([img, panel])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--device", default=default_device())
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--no-show", action="store_true", help="don't open the live view window (headless machines)")
    p.add_argument("--joystick", action="store_true", help="skip NoMaD; drive with the arrow keys (needs the window)")
    p.add_argument("--speed", type=float, default=0.5, help="joystick speed as a fraction of the robot's max v and w")
    p.add_argument("--hold-ms", type=int, default=400, help="joystick: an arrow key counts as held this long after its last press")
    args = p.parse_args()
    show = not args.no_show
    if args.joystick and not show:
        raise SystemExit("--joystick needs the window; drop --no-show")
    if not 0 < args.speed <= 1:
        raise SystemExit("--speed must be in (0, 1]")

    explorer = None
    if not args.joystick:
        if not os.path.exists(WEIGHTS):
            raise SystemExit(f"Missing weights: {WEIGHTS}")
        explorer = NomadExplorer(args.device, args.num_samples)
    server = InferenceServer(args.port)
    keys = KeyPoller(args.hold_ms / 1000)
    if show:  # create the window up front so it can take keyboard focus
        cv2.imshow(WINDOW, np.full((VIEW_SIZE, 2 * VIEW_SIZE, 3), 30, np.uint8))
    if args.joystick:
        print(f"JOYSTICK mode (no NoMaD): click the window, arrows drive, space/s stops, q quits. Waiting for frames on port {args.port}")
    else:
        print(f"NoMaD ready on {args.device}; waiting for frames on port {args.port}")

    try:
        while not keys.quit:
            # poll with a timeout while a window is open, so the window keeps repainting between frames
            frame = server.recv_frame(timeout_ms=100 if show else None)
            if show:
                keys.poll()
            if frame is None:
                continue
            if args.joystick:
                action = ActionMsg(frame.seq, frame.t_capture, velocity=keys.command(args.speed))
            else:
                try:
                    action = explorer.step(frame)
                except Exception as e:  # always answer, or the REQ/REP pair deadlocks
                    action = ActionMsg(frame.seq, frame.t_capture, status="error", error=repr(e))
            server.send_action(action)
            print(f"seq {frame.seq}: {action.status} infer {action.infer_ms:.1f} ms spread {action.spread:.3f} "
                  f"vel {action.velocity} {action.error}")
            if show:
                cv2.imshow(WINDOW, render(frame, action))
                keys.poll()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
