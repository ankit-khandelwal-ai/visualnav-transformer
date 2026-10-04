"""Inference server: receives frames from robot.py, runs NoMaD (exploration mode), returns waypoints.

On the Mac (needs PYTHONPATH to include the diffusion_policy repo, see latency_bench.py):
    python server.py                  # picks mps/cuda/cpu automatically
    python server.py --num-samples 8 --device mps
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
    args = p.parse_args()
    show = not args.no_show

    if not os.path.exists(WEIGHTS):
        raise SystemExit(f"Missing weights: {WEIGHTS}")
    explorer = NomadExplorer(args.device, args.num_samples)
    server = InferenceServer(args.port)
    print(f"NoMaD ready on {args.device}; waiting for frames on port {args.port}")

    try:
        while True:
            # poll with a timeout while a window is open, so the window keeps repainting between frames
            frame = server.recv_frame(timeout_ms=100 if show else None)
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue
            try:
                action = explorer.step(frame)
            except Exception as e:  # always answer, or the REQ/REP pair deadlocks
                action = ActionMsg(frame.seq, frame.t_capture, status="error", error=repr(e))
            server.send_action(action)
            print(f"seq {frame.seq}: {action.status} infer {action.infer_ms:.1f} ms spread {action.spread:.3f} {action.error}")
            if show:
                cv2.imshow(WINDOW, render(frame, action))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
