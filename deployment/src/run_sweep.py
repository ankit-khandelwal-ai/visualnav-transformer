"""Sweep: N exploration runs in the same scene from different start locations.

Runs sim_robot's control loop from K fixed, well-separated starts (same floor, same
navmesh island, >=SEPARATION_M apart, >=CLEARANCE_M from walls) and reports how
starting location affects exploration coverage:

    python run_sweep.py --host localhost --runs 5 --steps 400 --out ../sim_out/sweep1
    python run_sweep.py --host localhost --analyze-only --out ../sim_out/sweep1

Per-run artifacts in <out>/run0..runK/ (same files sim_robot.py writes), plus
<out>/sweep_summary.json and <out>/sweep_map.png (all 5 trajectories on one map).

The server must be running (server.py --no-show). NOTE: the server keeps its frame
context across runs (no reset), so the first ~3 steps of each run see the previous
run's tail — keep that in mind for very short runs.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "train"))

import habitat_sim  # noqa: E402
import magnum as mn  # noqa: E402
from habitat_sim.utils import common as U  # noqa: E402

from protocol import DEFAULT_PORT, FrameMsg, RobotClient  # noqa: E402
from robot import DT, RobotControl, preprocess_frame  # noqa: E402
from explore_metrics import augment_summary, write_coverage_csv  # noqa: E402
from sim_robot import (  # noqa: E402
    CoverageTracker, HabitatVirtualRobot, draw_coverage_map, draw_map, render_topdown,
)
from sim_robot import STUCK_MOVE_M, STUCK_STEPS  # noqa: E402

DEFAULT_SCENE = os.path.expanduser(
    "~/repositories/habitat_data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
)
STARTS_FILE = "start_points.json"

# fixed ground-floor starts (castle), all island 1, >=6 m apart, >=0.7 m clearance
FIXED_STARTS = [
    [0.58, 0.10, 9.29],
    [3.41, 0.10, 20.31],
    [-5.23, 0.21, 15.91],
    [-6.76, 0.13, 8.93],
    [4.22, 0.21, 14.13],
]

# fixed starts for HM3D scene 00770-NBg5UqG3di3 (from hm3d_sweep1/start_points.json),
# used instead of FIXED_STARTS when --scene points at that scene
HM3D_00770_STARTS = [
    [4.918, 0.107, 6.077],
    [-21.596, 0.107, 9.098],
    [-15.569, 0.107, 9.986],
    [-2.703, 0.107, 8.501],
    [-23.460, 0.107, 2.828],
]

SEPARATION_M = 6.0
CLEARANCE_M = 0.5


def sample_starts(pathfinder, n: int, seed: int = 0) -> list:
    """N navigable points on one floor, one island, separated, with clearance."""
    pathfinder.seed(seed)
    pts = np.array([np.array(pathfinder.get_random_navigable_point())
                    for _ in range(4000)])
    clear = np.array([pathfinder.distance_to_closest_obstacle(p) for p in pts])
    island = np.array([pathfinder.get_island(p) for p in pts])
    # anchor floor+island to the roomiest clear point
    anchor = pts[(clear >= CLEARANCE_M)][np.argmax(clear[(clear >= CLEARANCE_M)])]
    cands = pts[(np.abs(pts[:, 1] - anchor[1]) < 0.15)
                & (island == pathfinder.get_island(anchor))
                & (clear >= CLEARANCE_M)]
    sel = [anchor]
    for p in cands:
        if all(np.linalg.norm((p - q)[[0, 2]]) > SEPARATION_M for q in sel):
            sel.append(p)
        if len(sel) >= n:
            break
    return sel


def run_one(sim, args, start: np.ndarray, out_dir: str, run_idx: int) -> dict:
    """One exploration run from `start`; returns the summary dict."""
    robot = HabitatVirtualRobot(sim, radius=args.agent_radius,
                                allow_sliding=args.allow_sliding)
    robot.set_state(start, [0.0, 0.0, 0.0, 1.0])

    map_img, bounds, navmask = render_topdown(sim.pathfinder, height=float(start[1]))
    coverage = CoverageTracker(navmask, bounds, args.agent_radius)
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "episode_log.jsonl"), "w")
    import imageio
    video_writer = imageio.get_writer(os.path.join(out_dir, "episode.mp4"), fps=4)

    client = RobotClient(args.host, args.port, args.timeout_ms)
    positions, collision_flags, infer_times, timeouts = [], [], [], 0
    stuck_streak = 0
    goal = start  # exploration: goal unused; keeps summary fields populated

    try:
        print(f"\n=== run {run_idx}: start {np.round(start, 2).tolist()} ===")
        for seq in range(args.steps):
            t_step = time.monotonic()
            obs = sim.get_sensor_observations()
            rgb = np.asarray(obs["rgb"])[:, :, :3]
            video_writer.append_data(rgb)
            frame = FrameMsg(seq=seq, t_capture=time.time(),
                             image=preprocess_frame(rgb[:, :, ::-1].copy()))
            t_send = time.monotonic()
            action = client.request(frame)
            rtt_ms = (time.monotonic() - t_send) * 1000

            if action is None:
                v = w = 0.0
                status, timeouts = "timeout", timeouts + 1
            elif action.status != "ok" or action.waypoints is None:
                v = w = 0.0
                status = action.status + (f": {action.error}" if action.error else "")
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

            rec = {"seq": seq, "status": status, "rtt_ms": round(rtt_ms, 1),
                   "infer_ms": None if action is None else round(action.infer_ms, 1),
                   "v": round(v, 3), "w": round(w, 3), "collided": bool(collided),
                   "moved_m": round(moved, 4),
                   "position": [round(float(c), 3) for c in positions[-1]]}
            print(json.dumps(rec, default=float))
            log_file.write(json.dumps(rec, default=float) + "\n")
            log_file.flush()
            stuck_streak = stuck_streak + 1 if moved < STUCK_MOVE_M else 0
            if stuck_streak >= STUCK_STEPS:
                print(f"stuck for {stuck_streak} steps; terminating run")
                break
            time.sleep(max(0.0, DT - (time.monotonic() - t_step)))
    finally:
        client.close()
        log_file.close()
        video_writer.close()

    # same summary math as sim_robot.py
    path = np.array([start] + positions, dtype=np.float64)
    distance_m = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    flags = collision_flags
    events = int(flags[0]) + sum(1 for a, b in zip(flags, flags[1:]) if b and not a)
    first = next((i for i, f in enumerate(flags) if f), None)
    before_first = (round(float(np.linalg.norm(
        np.diff(path[:first + 2], axis=0), axis=1).sum()), 2) if first is not None else None)
    cov = coverage.coverage
    summary = {
        "run": run_idx, "start": np.round(start, 2).tolist(),
        "steps": len(positions), "distance_m": round(distance_m, 2),
        "coverage": None if cov is None else round(cov, 4),
        "collision_events": events, "collisions": int(sum(flags)),
        "collision_events_per_m": round(events / distance_m, 3) if distance_m > 0 else None,
        "meters_to_first_collision": before_first,
        "mean_infer_ms": round(float(np.mean(infer_times)), 1) if infer_times else None,
        "timeouts": timeouts,
        "terminated_stuck": stuck_streak >= STUCK_STEPS,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    draw_map(os.path.join(out_dir, "trajectory_map.png"), map_img, bounds,
             positions, goal, [p for p, f in zip(positions, flags) if f])
    draw_coverage_map(os.path.join(out_dir, "coverage_map.png"),
                      map_img, coverage.visited, navmask)
    # exploration metrics on top of the base summary (goal unused in sweeps)
    if coverage.curve:
        write_coverage_csv(out_dir, coverage.curve)
        augment_summary(summary, np.array([start] + positions), flags,
                       coverage.curve, args.steps, None, None)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
    return summary


def draw_sweep_map(out_path, map_img, bounds, starts, paths, mpp=0.05):
    """All runs on one map: each trajectory its own color, starts as filled dots."""
    from PIL import Image, ImageDraw
    from sim_robot import world_to_grid

    colors = [(230, 25, 75), (255, 140, 0), (60, 180, 75), (0, 130, 200), (130, 0, 200)]
    img = Image.fromarray(map_img.copy())
    d = ImageDraw.Draw(img)
    for i, (s, p) in enumerate(zip(starts, paths)):
        col = colors[i % len(colors)]
        pts = [world_to_grid(q[0], q[2], bounds, mpp)[::-1] for q in p]
        if len(pts) >= 2:
            d.line(pts, fill=col, width=2)
        sx, sy = world_to_grid(s[0], s[2], bounds, mpp)[::-1]
        d.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=col)
    img.save(out_path)


def analyze(out_dir: str) -> dict:
    """Aggregate run summaries into the sweep report."""
    runs = []
    for i in range(100):
        p = os.path.join(out_dir, f"run{i}", "summary.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            runs.append(json.load(f))
    if not runs:
        raise SystemExit(f"no run summaries under {out_dir}")
    covs = [r["coverage"] for r in runs if r["coverage"] is not None]
    evs = [r["collision_events"] for r in runs]
    dith = [r["heading_dither_rad_per_step"] for r in runs
            if r.get("heading_dither_rad_per_step") is not None]
    spnc = [r["steps_per_new_cell"] for r in runs
            if r.get("steps_per_new_cell") is not None]
    fails = [r.get("failure_mode", "?") for r in runs]

    def _mean(xs):
        return round(float(np.mean(xs)), 3) if xs else None

    report = {
        "n_runs": len(runs),
        "coverage_mean": round(float(np.mean(covs)), 4) if covs else None,
        "coverage_std": round(float(np.std(covs)), 4) if covs else None,
        "coverage_min": round(float(min(covs)), 4) if covs else None,
        "coverage_max": round(float(max(covs)), 4) if covs else None,
        "collision_events_total": int(sum(evs)),
        # exploration quality: low dither + low steps-per-new-cell = decisive explorer
        "heading_dither_mean": _mean(dith),
        "steps_per_new_cell_mean": _mean(spnc),
        "failure_modes": {m: fails.count(m) for m in set(fails)},
        "runs": runs,
    }
    with open(os.path.join(out_dir, "sweep_summary.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--steps", type=int, default=400, help="steps per run (400 = 100 s at 4 Hz)")
    p.add_argument("--timeout-ms", type=int, default=2000)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--waypoint-idx", type=int, default=2)
    p.add_argument("--agent-radius", type=float, default=0.18)
    p.add_argument("--allow-sliding", action="store_true",
                   help="let the agent slide along obstacles on collision (default: no)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-starts", action="store_true",
                   help="sample starts per the separation/clearance rules instead of the fixed set")
    p.add_argument("--out", required=True)
    p.add_argument("--analyze-only", action="store_true",
                   help="just re-aggregate existing runs and print the report")
    args = p.parse_args()

    if args.analyze_only:
        print(json.dumps(analyze(args.out), indent=1))
        return

    os.makedirs(args.out, exist_ok=True)

    # ---- sim (one Simulator, reused across runs; scenes are cheap to reload) ----
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = args.scene
    backend_cfg.random_seed = args.seed
    backend_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 0.75  # small ground robot; also navmesh clearance (overhangs above pass)
    agent_cfg.radius = args.agent_radius
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "rgb"; cam.resolution = [480, 640]
    cam.position = mn.Vector3(0.0, 0.65, 0.0)
    cam.hfov = 90
    agent_cfg.sensor_specifications = [cam]
    agent_cfg.action_space = {}
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    navmesh_settings = habitat_sim.nav.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = args.agent_radius
    navmesh_settings.agent_height = 0.75
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    fixed = (HM3D_00770_STARTS if "00770-NBg5UqG3di3" in os.path.basename(args.scene)
             else FIXED_STARTS)
    starts = (sample_starts(sim.pathfinder, args.runs, args.seed) if args.random_starts
              else [np.array(s) for s in fixed[:args.runs]])
    if len(starts) < args.runs:
        raise SystemExit(f"only {len(starts)} valid starts found; lower --runs or --random-starts rules")
    with open(os.path.join(args.out, STARTS_FILE), "w") as f:
        json.dump({"scene": args.scene, "starts": [np.round(s, 3).tolist() for s in starts]}, f, indent=1)

    map_img, bounds, _ = render_topdown(sim.pathfinder, height=float(starts[0][1]))
    summaries, all_paths = [], []
    try:
        for i, s in enumerate(starts):
            summaries.append(run_one(sim, args, s, os.path.join(args.out, f"run{i}"), i))
            # reread positions for the combined map (run_one only returns the summary)
            pos = []
            with open(os.path.join(args.out, f"run{i}", "episode_log.jsonl")) as f:
                for line in f:
                    pos.append(json.loads(line).get("position"))
            all_paths.append([p for p in pos if p is not None])
    finally:
        sim.close()

    draw_sweep_map(os.path.join(args.out, "sweep_map.png"), map_img, bounds,
                   starts, all_paths)
    report = analyze(args.out)
    print("\n=== SWEEP REPORT ===")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()