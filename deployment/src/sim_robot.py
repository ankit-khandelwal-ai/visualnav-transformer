"""Sim harness: habitat-sim camera -> NoMaD server -> velocity control, with logging.

Replaces robot.py (the physical client) with a habitat-sim "virtual robot", exercising
the exact same server/client code path as on the real robot:

    python server.py --device cuda                          # inference server
    python sim_robot.py --host localhost --out ../sim_out   # virtual robot

Each step: render RGB -> preprocess_frame() (same as robot.py) -> RobotClient.request()
-> waypoint_to_vel() (same as robot.py) -> velocity-integrate the habitat agent with
collision detection identical to habitat-lab's VelocityAction (navmesh snap without
sliding, moved-distance comparison). One JSON line is logged per step.

Exploration metrics in summary.json: coverage (fraction of the navigable floor
touched by the agent's disc), collision events (consecutive flagged steps deduped
into one event), collisions per meter, and meters to first collision.

PointNav episodes from habitat-lab provide start/goal (the goal is used only for
diagnostics; the exploration server masks it).
"""
import argparse
import gzip
import json
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))

from protocol import DEFAULT_PORT, FrameMsg, RobotClient  # noqa: E402
from robot import DT, RATE, RobotControl, preprocess_frame  # noqa: E402
from explore_metrics import augment_summary, write_coverage_csv  # noqa: E402

import habitat_sim  # noqa: E402
import magnum as mn  # noqa: E402
from habitat_sim import RigidState  # noqa: E402
from habitat_sim.physics import VelocityControl  # noqa: E402
from habitat_sim.utils import common as U  # noqa: E402

STUCK_STEPS = 10  # consecutive blocked steps (2.5 s at 4 Hz) before a run is terminated
# A step counts as blocked on displacement, not on the collision flag: with
# allow_sliding the agent keeps wall-following at ~0.1-0.2 m/step while collided=True,
# which is productive exploration, not being stuck. 0.02 m = 0.08 m/s at 4 Hz, vs the
# 0.25 m free-space step at MAX_V.
STUCK_MOVE_M = 0.02

HABITAT_LAB_ROOT = os.path.expanduser("~/repositories/habitat-lab")
DEFAULT_SCENES_DIR = os.path.expanduser(
    "~/repositories/habitat_data/scene_datasets/habitat-test-scenes"
)


def resolve_scene(scene_id: str) -> str:
    """Episode files reference scenes relative to the habitat-lab checkout; resolve."""
    candidates = [
        scene_id,
        os.path.join(HABITAT_LAB_ROOT, scene_id),
        os.path.join(DEFAULT_SCENES_DIR, os.path.basename(scene_id)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise SystemExit(f"scene not found: {scene_id} (tried: {candidates})")


def load_episode(episode_path: str, episode_idx: int) -> dict:
    with gzip.open(episode_path, "rt") as f:
        episodes = json.load(f)["episodes"]
    if episode_idx >= len(episodes):
        raise SystemExit(f"episode {episode_idx} out of range ({len(episodes)} available)")
    return episodes[episode_idx]


class HabitatVirtualRobot:
    """Habitat agent accepting (v, w) like the real robot, reporting collisions.

    Mirrors habitat-lab's VelocityAction (habitat/tasks/nav/nav.py): integrate the
    rigid state over dt, snap to the navmesh with try_step_no_sliding, and flag a
    collision when the filtered move is shorter than the commanded move.
    """

    def __init__(self, sim: habitat_sim.Simulator, radius: float = 0.18,
                 allow_sliding: bool = False):
        self.sim = sim
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self.allow_sliding = allow_sliding
        self.collisions = 0

    def set_state(self, position, rotation_coeffs):
        state = habitat_sim.AgentState(
            position=position, rotation=U.quat_from_coeffs(np.array(rotation_coeffs))
        )
        self.sim.get_agent(0).set_state(state, reset_sensors=True)

    def get_state(self):
        return self.sim.get_agent(0).get_state()

    def act(self, v: float, w: float, dt: float) -> Tuple[bool, float]:
        """Apply (v, w) for dt seconds. Returns (collided, distance_moved)."""
        self.vel_control.linear_velocity = np.array([0.0, 0.0, -v])
        self.vel_control.angular_velocity = np.array([0.0, w, 0.0])
        st = self.get_state()
        cur = RigidState(mn.Quaternion(st.rotation.imag, st.rotation.real), st.position)
        goal = self.vel_control.integrate_transform(dt, cur)
        step_fn = (self.sim.pathfinder.try_step if self.allow_sliding
                   else self.sim.pathfinder.try_step_no_sliding)
        final = step_fn(st.position, goal.translation)
        d_before = (goal.translation - st.position).dot()
        d_after = (final - st.position).dot()
        collided = (d_after + 1e-5) < d_before
        if collided:
            self.collisions += 1
        rot_coeffs = [*goal.rotation.vector, goal.rotation.scalar]
        self.set_state(final, rot_coeffs)
        return collided, math.sqrt(max(d_after, 0.0))


def render_topdown(pathfinder, height: float, meters_per_pixel: float = 0.05):
    """Greyscale top-down map of the navmesh slice at `height` (the agent's floor).
    Returns (img_HWC_uint8, bounds, navmask) with navmask True = navigable."""
    bounds = pathfinder.get_bounds()
    mask = pathfinder.get_topdown_view(meters_per_pixel, height)
    img = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2).astype(np.uint8) * 255  # floor white, walls black
    return img, bounds, mask


class CoverageTracker:
    """Fraction of the navigable floor (spawn-height slice) touched by the agent's disc.

    A cell is visited when the agent (a disc of `agent_radius`) passes over it;
    coverage = visited∩navigable / navigable. Only the spawn-height navmesh slice is
    counted, so multi-floor scenes under-report upper floors.
    """

    def __init__(self, navmask: np.ndarray, bounds, agent_radius: float, mpp: float = 0.05):
        self.navmask = navmask
        self.bounds = bounds
        self.mpp = mpp
        self.visited = np.zeros_like(navmask, dtype=bool)
        self.navigable_cells = int(navmask.sum())
        pad = max(1, int(round(agent_radius / mpp)))
        yy, xx = np.ogrid[-pad:pad + 1, -pad:pad + 1]
        self.struct = (xx * xx + yy * yy) <= pad * pad  # disc of ~agent_radius
        self.curve: List[float] = []  # coverage after each add(), for the coverage curve

    def add(self, position) -> None:
        r, c = world_to_grid(position[0], position[2], self.bounds, self.mpp)
        H, W = self.navmask.shape
        if not (0 <= r < H and 0 <= c < W):
            self.curve.append(self.coverage)  # off-map step: record flat, skip the disc
            return
        marker = np.zeros((H, W), dtype=bool)
        marker[r, c] = True
        self.visited |= binary_dilation(marker, structure=self.struct)
        self.curve.append(self.coverage)

    @property
    def coverage(self) -> Optional[float]:
        if self.navigable_cells == 0:
            return None
        return float((self.visited & self.navmask).sum()) / self.navigable_cells


def draw_coverage_map(out_path, map_img, visited, navmask):
    """Same top-down map with visited navigable cells painted green."""
    from PIL import Image

    arr = map_img.copy()
    arr[visited & navmask] = (120, 220, 120)
    Image.fromarray(arr).save(out_path)


def world_to_grid(x: float, z: float, bounds, mpp: float) -> Tuple[int, int]:
    """habitat.utils.visualizations.maps.to_grid convention: row from z, col from x."""
    lower, _ = bounds
    return int((z - lower[2]) / mpp), int((x - lower[0]) / mpp)


def draw_map(out_path, map_img, bounds, positions, goal, collision_positions, mpp=0.05):
    from PIL import Image, ImageDraw

    img = Image.fromarray(map_img)
    d = ImageDraw.Draw(img)
    # PIL wants (x=col, y=row); our grid points are (row, col)
    pts = [world_to_grid(p[0], p[2], bounds, mpp)[::-1] for p in positions]
    if len(pts) >= 2:
        d.line(pts, fill=(255, 0, 0), width=2)
    for gx, gy in [world_to_grid(p[0], p[2], bounds, mpp)[::-1]
                   for p in collision_positions]:
        d.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], fill=(255, 140, 0))
    if len(pts) >= 1:
        sx, sy = pts[0]
        d.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], outline=(0, 200, 0), width=3)
    g = world_to_grid(goal[0], goal[2], bounds, mpp)[::-1]
    d.ellipse([g[0] - 5, g[1] - 5, g[0] + 5, g[1] + 5], outline=(0, 0, 255), width=3)
    img.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True, help="IP of the machine running server.py")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--scene", default=os.path.join(
        DEFAULT_SCENES_DIR, "skokloster-castle.glb"))
    p.add_argument("--episode", default=None,
                   help="pointnav episode .json.gz; overrides --scene with the episode's scene")
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--timeout-ms", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--sample", type=int, default=0,
                   help="which of the server's sampled trajectories to follow")
    p.add_argument("--waypoint-idx", type=int, default=2,
                   help="which predicted waypoint to steer toward (robot.py default: 2)")
    p.add_argument("--out", default=None, help="output dir for logs/map/video")
    p.add_argument("--save-frames", action="store_true", help="save every camera frame as PNG")
    p.add_argument("--save-video", action="store_true", help="save episode as MP4")
    p.add_argument("--agent-radius", type=float, default=0.18)
    p.add_argument("--agent-height", type=float, default=0.75,
                   help="body height; also the navmesh clearance height (overhangs above this are passable)")
    p.add_argument("--camera-height", type=float, default=0.65)
    p.add_argument("--camera-tilt", type=float, default=0.0, help="camera pitch, degrees")
    p.add_argument("--hfov", type=int, default=90)
    p.add_argument("--allow-sliding", action="store_true",
                   help="let the agent slide along obstacles on collision (default: no)")
    p.add_argument("--sim-gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sim: Optional[habitat_sim.Simulator] = None
    client: Optional[RobotClient] = None
    robot: Optional[HabitatVirtualRobot] = None
    log_file = None
    video_writer = None
    out_dir = None
    map_img = bounds = None
    navmask = None
    positions: List = []
    collision_positions: List = []
    collision_flags: List[bool] = []
    stuck_streak = 0
    goal = None
    infer_times: List[float] = []
    timeouts = 0
    coverage = None

    try:
        # ---------------- sim setup ----------------
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = (resolve_scene(load_episode(args.episode, args.episode_idx)["scene_id"])
                                if args.episode else resolve_scene(args.scene))
        backend_cfg.random_seed = args.seed
        backend_cfg.gpu_device_id = args.sim_gpu
        backend_cfg.enable_physics = False

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = args.agent_height
        agent_cfg.radius = args.agent_radius
        cam = habitat_sim.CameraSensorSpec()
        cam.uuid = "rgb"
        cam.resolution = [480, 640]
        cam.position = mn.Vector3(0.0, args.camera_height, 0.0)
        cam.orientation = mn.Vector3(math.radians(args.camera_tilt), 0.0, 0.0)
        cam.hfov = args.hfov
        agent_cfg.sensor_specifications = [cam]
        agent_cfg.action_space = {}

        sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
        navmesh_settings = habitat_sim.nav.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = args.agent_radius
        navmesh_settings.agent_height = args.agent_height
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

        # ---------------- episode start/goal ----------------
        if args.episode:
            ep = load_episode(args.episode, args.episode_idx)
            start_pos = np.array(ep["start_position"], dtype=np.float32)
            start_rot = ep["start_rotation"]
            goal = np.array(ep["goals"][0]["position"], dtype=np.float32)
        else:
            start_pos = sim.pathfinder.get_random_navigable_point()
            start_rot = [0.0, 0.0, 0.0, 1.0]
            goal = np.array(start_pos) + np.array([3.0, 0.0, 0.0])

        robot = HabitatVirtualRobot(sim, radius=args.agent_radius,
                                    allow_sliding=args.allow_sliding)
        robot.set_state(start_pos, start_rot)

        map_img, bounds, navmask = render_topdown(sim.pathfinder, height=float(start_pos[1]))
        coverage = CoverageTracker(navmask, bounds, args.agent_radius)

        if args.out:
            out_dir = args.out
            os.makedirs(out_dir, exist_ok=True)
            log_file = open(os.path.join(out_dir, "episode_log.jsonl"), "w")
            if args.save_video:
                import imageio
                video_writer = imageio.get_writer(
                    os.path.join(out_dir, "episode.mp4"), fps=RATE)

        client = RobotClient(args.host, args.port, args.timeout_ms)
        print(f"sim_robot: scene {backend_cfg.scene_id}")
        print(f"  start {np.round(start_pos, 2).tolist()}  goal {np.round(goal, 2).tolist()}"
              f"  ({args.max_steps} steps max, server {args.host}:{args.port})")

        # ---------------- control loop ----------------
        for seq in range(args.max_steps):
            t_step = time.monotonic()
            obs = sim.get_sensor_observations()
            rgb = np.asarray(obs["rgb"])[:, :, :3]
            if args.save_frames and out_dir:
                from PIL import Image
                Image.fromarray(rgb).save(os.path.join(out_dir, f"frame_{seq:04d}.png"))
            if video_writer is not None:
                video_writer.append_data(rgb)

            bgr = rgb[:, :, ::-1].copy()  # preprocess_frame expects BGR (cv2 convention)
            frame = FrameMsg(seq=seq, t_capture=time.time(), image=preprocess_frame(bgr))

            t_send = time.monotonic()
            action = client.request(frame)
            rtt_ms = (time.monotonic() - t_send) * 1000

            if action is None:
                v = w = 0.0
                status = "timeout"
                timeouts += 1
            elif action.status != "ok" or action.waypoints is None:
                v = w = 0.0
                status = f"{action.status}" + (f": {action.error}" if action.error else "")
            else:
                v, w = RobotControl.waypoint_to_vel(
                    action.waypoints[args.sample][args.waypoint_idx], args.waypoint_idx)
                status = "ok"
                infer_times.append(action.infer_ms)

            collided, moved = robot.act(v, w, DT)
            st = robot.get_state()
            positions.append([st.position[0], st.position[1], st.position[2]])
            collision_flags.append(collided)
            coverage.add(positions[-1])
            if collided:
                collision_positions.append(positions[-1])

            rec = {
                "seq": seq, "status": status, "rtt_ms": round(rtt_ms, 1),
                "loop_ms": round((time.monotonic() - t_step) * 1000, 1),
                "infer_ms": None if action is None else round(action.infer_ms, 1),
                "spread": None if action is None else round(action.spread, 4),
                "v": round(v, 3), "w": round(w, 3),
                "collided": bool(collided), "moved_m": round(moved, 4),
                "dist_to_goal": round(float(np.linalg.norm(
                    np.array([st.position[0], st.position[1], st.position[2]]) - goal)), 3),
            }
            print(json.dumps(rec))
            if log_file:
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
            # a run is terminated when the agent can't move for STUCK_STEPS straight steps
            stuck_streak = stuck_streak + 1 if moved < STUCK_MOVE_M else 0
            if stuck_streak >= STUCK_STEPS:
                print(f"stuck for {stuck_streak} steps; terminating run")
                break
            time.sleep(max(0.0, DT - (time.monotonic() - t_step)))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if client is not None:
            client.close()
        if log_file is not None:
            log_file.close()
        if video_writer is not None:
            video_writer.close()
        if sim is not None:
            sim.close()
        if out_dir and robot is not None and map_img is not None and positions:
            path = np.array([start_pos] + positions, dtype=np.float64)
            distance_m = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

            # dedupe consecutive collided steps into one event (wall-press at 4 Hz = 1 event)
            flags = [p is not None and p for p in collision_flags]
            events = int(flags[0]) + sum(1 for a, b in zip(flags, flags[1:]) if b and not a)
            first_collision_idx = next((i for i, f in enumerate(flags) if f), None)
            dist_before_first = (
                round(float(np.linalg.norm(np.diff(path[:first_collision_idx + 2], axis=0), axis=1).sum()), 2)
                if first_collision_idx is not None else None)

            summary = {
                "steps": len(positions), "collisions": robot.collisions,
                "collision_events": events,
                "collision_events_per_m": round(events / distance_m, 3) if distance_m > 0 else None,
                "meters_to_first_collision": dist_before_first,
                "coverage": (round(coverage.coverage, 4)
                             if coverage is not None and coverage.coverage is not None else None),
                "distance_m": round(distance_m, 2),
                "final_dist_to_goal": round(float(np.linalg.norm(path[-1] - goal)), 3),
                "mean_infer_ms": round(float(np.mean(infer_times)), 1) if infer_times else None,
                "timeouts": timeouts,
                "terminated_stuck": stuck_streak >= STUCK_STEPS,
            }
            with open(os.path.join(out_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=1)
            write_coverage_csv(out_dir, coverage.curve)
            draw_map(os.path.join(out_dir, "trajectory_map.png"),
                     map_img, bounds, positions, goal, collision_positions)
            if coverage is not None:
                draw_coverage_map(os.path.join(out_dir, "coverage_map.png"),
                                  map_img, coverage.visited, navmask)
            # exploration metrics (steps_per_new_cell, revisit ratio, dithering, ...)
            # computed from the full trajectory after the run ends
            if coverage is not None and coverage.curve:
                augment_summary(summary, np.array([start_pos] + positions),
                                collision_flags, coverage.curve,
                                args.max_steps, goal, sim.pathfinder)
                with open(os.path.join(out_dir, "summary.json"), "w") as f:
                    json.dump(summary, f, indent=1)
            print("summary:", json.dumps(summary))


if __name__ == "__main__":
    main()