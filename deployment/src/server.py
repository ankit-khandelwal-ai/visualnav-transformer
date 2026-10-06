"""Inference server: receives frames from robot.py, runs NoMaD (exploration mode), returns waypoints.

On the Mac (needs PYTHONPATH to include the diffusion_policy repo, see latency_bench.py):
    python server.py                  # picks mps/cuda/cpu automatically
    python server.py --num-samples 8 --device mps
    python server.py --joystick       # NoMaD still runs and is displayed, but the robot gets your arrow-key
                                      # commands instead (click the window first)
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

from latency_bench import NORM, build_model, sync, to_tensor
from protocol import DEFAULT_PORT, ActionMsg, FrameMsg, InferenceServer
from robot import DT, WAYPOINT_IDX, RobotControl

HERE = os.path.dirname(os.path.abspath(__file__))
NOMAD_CFG = os.path.join(HERE, "../../train/config/nomad.yaml")
DATA_CFG = os.path.join(HERE, "../../train/vint_train/data/data_config.yaml")
WEIGHTS = os.path.join(HERE, "../model_weights/nomad.pth")
TOPOMAP_IMAGES_DIR = os.path.join(HERE, "../topomaps/images")  # create_topomap.py convention
with open(os.path.join(HERE, "../config/robot.yaml")) as _f:
    _robot_cfg = yaml.safe_load(_f)
MAX_V, MAX_W = _robot_cfg["max_v"], _robot_cfg["max_w"]  # the robot clamps to these too


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


SELECT_MODES = ("first", "forward", "consistent", "blend")


class NomadExplorer:
    """Goal-masked NoMaD: keeps a context of recent frames and samples collision-free trajectories.

    The diffusion head returns `num_samples` independent modes per frame. Taking a fixed index
    means the robot follows a fresh random mode every step, which shows up as heading dither and
    looping. `select` ranks the samples and reorders them so the chosen one is waypoints[0] (what
    the clients follow and what render() draws in green); "first" keeps the old behaviour exactly.
    """

    def __init__(self, device: str, num_samples: int, select: str = "blend",
                 consistency_weight: float = 0.7, waypoint_idx: int = WAYPOINT_IDX):
        if select not in SELECT_MODES:
            raise ValueError(f"select must be one of {SELECT_MODES}")
        self.device = torch.device(device)
        self.num_samples = num_samples
        self.select = select
        self.consistency_weight = consistency_weight
        self.waypoint_idx = waypoint_idx
        self.prev_heading = None  # last chosen heading, rotated into the current frame
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
        if frame.seq == 0:  # clients restart seq at 0 for each run: don't leak the previous run's frames
            self.context.clear()
            self.prev_heading = None
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
        spread = float(waypoints.std(axis=0).mean())
        order = self.rank(waypoints)
        return ActionMsg(frame.seq, frame.t_capture, waypoints=waypoints[order].astype(np.float32),
                         infer_ms=(time.perf_counter() - t0) * 1000, spread=spread, chosen=int(order[0]))

    def rank(self, wp: np.ndarray) -> np.ndarray:
        """Order the samples best-first. wp is (S, T, 2), x forward and y left in the robot frame."""
        n = wp.shape[0]
        if self.select == "first" or n == 1:
            return np.arange(n)

        # heading of the waypoint the client actually steers at, and net forward reach
        heading = np.arctan2(wp[:, self.waypoint_idx, 1], wp[:, self.waypoint_idx, 0])
        reach = wp[:, -1, 0]
        span = float(reach.max() - reach.min())
        forward = (reach - reach.min()) / span if span > 1e-9 else np.full(n, 0.5)

        if self.prev_heading is None:  # first scored frame of a run: nothing to be consistent with
            consistent = np.full(n, 0.5)
        else:
            consistent = (1 + np.cos(heading - self.prev_heading)) / 2

        if self.select == "forward":
            score = forward
        elif self.select == "consistent":
            score = consistent
        else:
            score = (1 - self.consistency_weight) * forward + self.consistency_weight * consistent
        order = np.argsort(-score)

        # Reference heading for the next frame. The client turns to face its waypoint within DT but
        # clips at MAX_W, so after a large turn it only gets part of the way there; carry the
        # remainder. Use the client's own (clipped) formula so this tracks what it will really do.
        best = int(order[0])
        w_cmd = RobotControl.waypoint_to_vel(wp[best, self.waypoint_idx], self.waypoint_idx)[1]
        self.prev_heading = float(heading[best] - w_cmd * DT)
        return order


class NomadGoalNavigator:
    """Goal-conditioned NoMaD over a topomap, mirroring navigate.py's logic:

    each step, encode the current obs against the topomap nodes within `radius` of the
    last localized node (input_goal_mask=0: the goal branch is live), run dist_pred_net
    to find the closest node, aim at that node or the next one (within close_threshold),
    and condition the diffusion head on that subgoal encoding. Sample 0 is followed,
    like navigate.py (no explorer-style ranking: the goal image does the constraining).
    """

    def __init__(self, device: str, topomap_dir: str, goal_node: int,
                 radius: int = 4, close_threshold: float = 3.0):
        self.device = torch.device(device)
        self.radius = radius
        self.close_threshold = close_threshold
        self.cfg = yaml.safe_load(open(NOMAD_CFG))
        stats = yaml.safe_load(open(DATA_CFG))["action_stats"]
        self.stat_min, self.stat_max = np.array(stats["min"]), np.array(stats["max"])
        self.image_size = tuple(self.cfg["image_size"])
        self.num_iters = self.cfg["num_diffusion_iters"]
        self.model = build_model(self.cfg, WEIGHTS, self.device)
        self.sched = DDPMScheduler(num_train_timesteps=self.num_iters,
                                   beta_schedule="squaredcos_cap_v2",
                                   clip_sample=True, prediction_type="epsilon")
        self.context = collections.deque(maxlen=self.cfg["context_size"] + 1)
        # only the numbered .png nodes: the folder may also hold a preview .gif
        names = sorted([f for f in os.listdir(topomap_dir) if f.endswith(".png")],
                       key=lambda x: int(x.split(".")[0]))
        self.topomap = [Image.open(os.path.join(topomap_dir, n)).convert("RGB") for n in names]
        if not -1 <= goal_node < len(self.topomap):
            raise SystemExit(f"goal node {goal_node} out of range (0..{len(self.topomap) - 1})")
        self.goal_node = len(self.topomap) - 1 if goal_node == -1 else goal_node
        self.closest_node = 0
        self.reached = False
        self.subgoal_node = None  # node the diffusion head is conditioned on this step
        self.subgoal_img = None  # its topomap image, for the live view

    @torch.no_grad()
    def step(self, frame: FrameMsg) -> ActionMsg:
        if frame.seq == 0:  # clients restart seq at 0 for each run
            self.context.clear()
            self.closest_node = 0
            self.reached = False
        self.context.append(Image.fromarray(frame.image))
        if len(self.context) < self.context.maxlen:
            return ActionMsg(frame.seq, frame.t_capture, status="warmup")

        t0 = time.perf_counter()
        obs = to_tensor(list(self.context), self.image_size).to(self.device)
        start = max(self.closest_node - self.radius, 0)
        end = min(self.closest_node + self.radius + 1, self.goal_node)
        # goal nodes stack along the BATCH dim, one 3ch image each (to_tensor would
        # concat them along channels like a context; the model wants (N, 3, H, W))
        goal_batch = torch.stack([NORM(g.resize(self.image_size))
                                  for g in self.topomap[start:end + 1]]).to(self.device)
        # 0 = goal visible; one mask value per batch entry (navigate.py: mask.repeat(N))
        mask = torch.zeros(end - start + 1, dtype=torch.long, device=self.device)
        cond = self.model("vision_encoder", obs_img=obs.repeat(end - start + 1, 1, 1, 1),
                          goal_img=goal_batch, input_goal_mask=mask)
        dists = self.model("dist_pred_net", obsgoal_cond=cond)
        dists = dists.flatten().detach().cpu().numpy()
        print(f"node dists: {list(zip(range(start, end + 1), np.round(dists, 2).tolist()))}")
        min_idx = int(np.argmin(dists))
        self.closest_node = min_idx + start
        # navigate.py: when close to the best node, steer toward the NEXT one on the way to the goal
        sg_idx = min(min_idx + int(dists[min_idx] < self.close_threshold), len(cond) - 1)
        self.subgoal_node = start + sg_idx
        self.subgoal_img = self.topomap[self.subgoal_node]
        obs_cond = cond[sg_idx].unsqueeze(0)
        obs_cond = obs_cond.repeat(1, 1) if obs_cond.ndim == 2 else obs_cond.repeat(1, 1, 1)

        act = torch.randn((1, self.cfg["len_traj_pred"], 2), device=self.device)
        self.sched.set_timesteps(self.num_iters)
        for k in self.sched.timesteps:
            noise = self.model("noise_pred_net", sample=act, timestep=k, global_cond=obs_cond)
            act = self.sched.step(model_output=noise, timestep=k, sample=act).prev_sample
        sync(self.device)
        # same as train_utils.get_action: unnormalize deltas, then cumulative sum -> waypoints
        deltas = (act.cpu().numpy() + 1) / 2 * (self.stat_max - self.stat_min) + self.stat_min
        waypoints = np.cumsum(deltas, axis=1)
        self.reached = self.closest_node >= self.goal_node
        return ActionMsg(frame.seq, frame.t_capture,
                         waypoints=waypoints.astype(np.float32),
                         infer_ms=(time.perf_counter() - t0) * 1000,
                         spread=float(waypoints.std(axis=0).mean()), chosen=0,
                         closest_node=self.closest_node, reached_goal=self.reached)


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


def render(frame: FrameMsg, action: ActionMsg, subgoal_img=None, subgoal_node=None) -> np.ndarray:
    """Left: the frame the model saw, upscaled. Right: top-down view of the sampled trajectories
    (robot at the bottom, forward is up, left is left; the selected sample, which the server has
    reordered to index 0 and the robot follows, is green). In goal mode, the topomap image the
    model is steering toward is overlaid on the panel's top-right."""
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

    if subgoal_img is not None:  # goal mode: the topomap node the model is conditioned on
        TH, TW = 120, 120  # thumbnail size on the obs frame
        th = cv2.resize(np.asarray(subgoal_img), (TW, TH), interpolation=cv2.INTER_CUBIC)
        x0, y0 = VIEW_SIZE - TW - 8, 8  # top-right of the observation frame
        img[y0:y0 + TH, x0:x0 + TW] = cv2.cvtColor(th, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img, (x0, y0), (x0 + TW, y0 + TH), (255, 255, 255), 1)
        cv2.putText(img, f"goal node {subgoal_node}", (x0, y0 + TH + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    lines = [f"seq {frame.seq}  {action.status}", f"infer {action.infer_ms:.0f} ms  spread {action.spread:.2f}"]
    if action.closest_node is not None:
        lines.append(f"node {action.closest_node}" + ("  GOAL REACHED" if action.reached_goal else ""))
    if action.velocity is not None:
        lines.append(f"JOYSTICK  v {action.velocity[0]:+.2f} m/s  w {action.velocity[1]:+.2f} rad/s")
    for i, text in enumerate(lines):
        cv2.putText(img, text, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return np.hstack([img, panel])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--device", default=default_device())
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--select", default="blend", choices=SELECT_MODES,
                   help="how to pick which of the --num-samples trajectories the robot follows: "
                        "'first' = sample 0 (old behaviour), 'forward' = furthest reach, "
                        "'consistent' = closest heading to the previous choice, 'blend' = both")
    p.add_argument("--consistency-weight", type=float, default=0.7,
                   help="--select blend: weight on consistency vs forward reach, in [0, 1]")
    p.add_argument("--waypoint-idx", type=int, default=WAYPOINT_IDX,
                   help="waypoint the scorer reads the heading from; must match the client's --waypoint-idx")
    p.add_argument("--no-show", action="store_true", help="don't open the live view window (headless machines)")
    p.add_argument("--record-frames", action="store_true",
                   help="save the annotated view to a video instead of showing it (needs --out)")
    p.add_argument("--joystick", action="store_true",
                   help="send arrow-key commands to the robot instead of NoMaD's (NoMaD still runs and is displayed)")
    p.add_argument("--speed", type=float, default=0.5, help="joystick speed as a fraction of the robot's max v and w")
    p.add_argument("--hold-ms", type=int, default=400, help="joystick: an arrow key counts as held this long after its last press")
    p.add_argument("--out", default=None, help="directory for --record-frames output (created if missing)")
    p.add_argument("--record-topomap", action="store_true",
                   help="with --joystick: save topomap node images while you drive (needs --topomap-dir)")
    p.add_argument("--topomap-dir", default=None,
                   help="topomap name; images go to ../topomaps/images/<name>/ (create_topomap.py convention)")
    p.add_argument("--topomap-dt", type=float, default=1.0,
                   help="seconds between saved topomap nodes (default 1.0, like create_topomap.py --dt)")
    p.add_argument("--goal-topomap-dir", default=None,
                   help="goal navigation: topomap name under ../topomaps/images/ (like --topomap-dir)")
    p.add_argument("--goal-node", type=int, default=-1,
                   help="goal navigation: goal node index, -1 = the last node (default -1)")
    p.add_argument("--goal-radius", type=int, default=4,
                   help="goal navigation: how many nodes around the last localized one to match against (default 4)")
    p.add_argument("--goal-close-threshold", type=float, default=3.0,
                   help="goal navigation: dist_pred_net distance below which the next node becomes the subgoal (default 3)")
    args = p.parse_args()
    show = not args.no_show and not args.record_frames
    record = args.record_frames
    if args.joystick and not show:
        raise SystemExit("--joystick needs the window; drop --no-show and --record-frames")
    if args.record_topomap and not args.joystick:
        raise SystemExit("--record-topomap is for joystick mode: you drive, it records")
    if args.record_topomap and not args.topomap_dir:
        raise SystemExit("--record-topomap needs --topomap-dir")
    if args.goal_topomap_dir and args.joystick:
        raise SystemExit("pick one: --joystick (you drive) or --goal-topomap-dir (NoMaD navigates)")
    if args.goal_topomap_dir and args.record_topomap:
        raise SystemExit("pick one: --record-topomap (you drive, it records) or --goal-topomap-dir")
    if args.topomap_dt <= 0:
        raise SystemExit("--topomap-dt must be positive")
    if record and not args.out:
        raise SystemExit("--record-frames needs --out")
    if not 0 < args.speed <= 1:
        raise SystemExit("--speed must be in (0, 1]")
    if not 0 <= args.consistency_weight <= 1:
        raise SystemExit("--consistency-weight must be in [0, 1]")

    if not os.path.exists(WEIGHTS):
        raise SystemExit(f"Missing weights: {WEIGHTS}")
    if args.goal_topomap_dir:  # goal navigation: NomadGoalNavigator instead of the explorer
        goal_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.goal_topomap_dir)
        if not os.path.isdir(goal_dir):
            raise SystemExit(f"topomap not found: {goal_dir}")
        explorer = NomadGoalNavigator(args.device, goal_dir, args.goal_node,
                                      args.goal_radius, args.goal_close_threshold)
        print(f"GOAL NAVIGATION: {len(explorer.topomap)} nodes, goal node {explorer.goal_node}, "
              f"radius {args.goal_radius}, close-threshold {args.goal_close_threshold}")
    else:
        explorer = NomadExplorer(args.device, args.num_samples, args.select,
                                args.consistency_weight, args.waypoint_idx)
    server = InferenceServer(args.port)
    keys = KeyPoller(args.hold_ms / 1000)
    writer = None
    topo = None  # (dir, dt, next_save_t, count) for --record-topomap
    if record:
        os.makedirs(args.out, exist_ok=True)
        import imageio
        writer = imageio.get_writer(os.path.join(args.out, "server_view.mp4"), fps=_robot_cfg["frame_rate"])
        print(f"recording server view to {os.path.join(args.out, 'server_view.mp4')}")
    if args.record_topomap:
        topo_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.topomap_dir)
        if os.path.isdir(topo_dir):
            raise SystemExit(f"{topo_dir} already exists; remove it or pick another --topomap-dir")
        os.makedirs(topo_dir)
        topo = {"dir": topo_dir, "dt": args.topomap_dt, "next": None, "count": 0}
        print(f"recording topomap to {topo_dir} (one node every {args.topomap_dt} s)")
    if show:  # create the window up front so it can take keyboard focus
        cv2.imshow(WINDOW, np.full((VIEW_SIZE, 2 * VIEW_SIZE, 3), 30, np.uint8))
    sel = args.select + (f" (consistency {args.consistency_weight:.2f})" if args.select == "blend" else "")
    print(f"NoMaD ready on {args.device}; {args.num_samples} samples, select={sel}, "
          f"waypoint-idx={args.waypoint_idx}; waiting for frames on port {args.port}")
    if args.joystick:
        print("JOYSTICK: the robot gets your commands, not NoMaD's. Click the window; arrows drive, space/s stops, q quits.")

    try:
        while not keys.quit:
            # poll with a timeout while a window is open, so the window keeps repainting between frames
            frame = server.recv_frame(timeout_ms=100 if show else None)
            if show:
                keys.poll()
            if frame is None:
                continue
            try:
                action = explorer.step(frame)
            except Exception as e:  # always answer, or the REQ/REP pair deadlocks
                action = ActionMsg(frame.seq, frame.t_capture, status="error", error=repr(e))
            if args.joystick:  # NoMaD's output is only displayed; the robot follows the keys
                action.velocity = keys.command(args.speed)
            server.send_action(action)
            print(f"seq {frame.seq}: {action.status} infer {action.infer_ms:.1f} ms spread {action.spread:.3f} "
                  f"chosen {action.chosen} vel {action.velocity}"
                  + (f" node {action.closest_node}" if action.closest_node is not None else "")
                  + (f" GOAL-REACHED" if action.reached_goal else "")
                  + (f" {action.error}" if action.error else ""))
            if show:
                cv2.imshow(WINDOW, render(frame, action, getattr(explorer, "subgoal_img", None),
                                          getattr(explorer, "subgoal_node", None)))
                keys.poll()
            if topo is not None and (topo["next"] is None or frame.t_capture >= topo["next"]):
                # topomap node: the preprocessed frame the model sees, saved like create_topomap.py
                # (node 0 at the first frame, then one every topomap-dt s of frame timestamps)
                from PIL import Image as PILImage
                PILImage.fromarray(frame.image).save(os.path.join(topo["dir"], f"{topo['count']}.png"))
                topo["next"] = frame.t_capture + topo["dt"]
                topo["count"] += 1
                print(f"topomap node {topo['count'] - 1} saved")
            if writer is not None:
                writer.append_data(cv2.cvtColor(
                    render(frame, action, getattr(explorer, "subgoal_img", None),
                           getattr(explorer, "subgoal_node", None)), cv2.COLOR_BGR2RGB))
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        if writer is not None:
            writer.close()
        if topo is not None:
            print(f"topomap done: {topo['count']} nodes in {topo['dir']}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
