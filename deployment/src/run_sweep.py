"""Goal-navigation sweep: N trials per topomap, a fresh server + sim_robot per trial.

    python run_sweep.py --trials 5
    python run_sweep.py --trials 5 --sweep 7          # write to ../sim_out/sweep7
    python run_sweep.py --analyze-only --sweep 7      # just re-aggregate

For each topomap in --topomaps (default: tv_to_room_sparse, then fridge_to_room_sparse)
and each trial, this:

  1. starts `server.py --goal-topomap-dir <map> --goal-radius 2 --record-frames --out <trial dir>`
     and waits for its "NoMaD ready" line,
  2. runs `sim_robot.py --host localhost --scene ../scenes/apartment.glb --start <map's start>
     --yaw 0.0 --max-steps 150 --allow-sliding --out <trial dir> --save-video` to completion,
  3. SIGINTs the server (so it closes server_view.mp4) and waits for it to exit.

Output layout:

    ../sim_out/sweep<N>/<topomap>/trial<i>/   episode_log.jsonl, summary.json, trajectory_map.png,
                                              coverage_map.png, episode.mp4 (sim) +
                                              server_view.mp4, server_log.txt (server) + sim_log.txt
    ../sim_out/sweep<N>/sweep_summary.json    per-map aggregate + every trial summary

--sweep defaults to one past the highest existing ../sim_out/sweep<N>.
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_OUT = os.path.join(HERE, "..", "sim_out")
DEFAULT_SCENE = os.path.join(HERE, "..", "scenes", "apartment.glb")
# server.py imports diffusion_policy, which is a bare checkout (not pip-installed) here
DEFAULT_DIFFUSION_POLICY_DIR = os.path.expanduser("~/repositories/diffusion_policy")

# per-topomap start pose for sim_robot.py (--yaw is 0.0 for all)
STARTS = {
    "tv_to_room_sparse": [-2.258, -0.951, 2.049],
    "fridge_to_room_sparse": [-3.2, -0.951, -1.298],
}
READY_LINE = "NoMaD ready"
SERVER_READY_TIMEOUT_S = 300  # model load + first-time CUDA init can be slow
SERVER_EXIT_TIMEOUT_S = 30


def next_sweep_number() -> int:
    nums = [int(m.group(1)) for d in glob.glob(os.path.join(SIM_OUT, "sweep*"))
            if (m := re.fullmatch(r"sweep(\d+)", os.path.basename(d)))]
    return max(nums, default=0) + 1


def wait_for_line(path: str, needle: str, proc: subprocess.Popen, timeout_s: float) -> bool:
    """Poll `path` until it contains `needle`; False if the process dies or times out."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if os.path.exists(path):
            with open(path, errors="replace") as f:
                if needle in f.read():
                    return True
        if proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def stop_server(proc: subprocess.Popen):
    """SIGINT first: server.py catches KeyboardInterrupt and finalizes its recording."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=SERVER_EXIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print("  server did not exit after SIGINT; killing")
        proc.kill()
        proc.wait()


def run_trial(args, topomap: str, trial: int, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    server_log = os.path.join(out_dir, "server_log.txt")
    sim_log = os.path.join(out_dir, "sim_log.txt")

    server_cmd = [args.python, os.path.join(HERE, "server.py"), "--port", str(args.port),
                  *(["--device", args.device] if args.device else []),
                  "--goal-topomap-dir", topomap, "--goal-radius", str(args.goal_radius),
                  "--record-frames", "--out", out_dir] + args.server_args
    sim_cmd = [args.python, os.path.join(HERE, "sim_robot.py"),
               "--host", "localhost", "--port", str(args.port), "--scene", args.scene,
               "--start", *[str(c) for c in STARTS[topomap]], "--yaw", "0.0",
               "--max-steps", str(args.max_steps), "--allow-sliding",
               "--out", out_dir, "--save-video"] + args.sim_args

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"  # stdout goes to a file; we poll it for the ready line
    if args.diffusion_policy_dir:
        env["PYTHONPATH"] = os.pathsep.join(
            [args.diffusion_policy_dir] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])

    print(f"\n=== {topomap} trial {trial} -> {out_dir}")
    print("  server:", " ".join(server_cmd))
    with open(server_log, "w") as slog:
        server = subprocess.Popen(server_cmd, cwd=HERE, env=env, stdout=slog, stderr=subprocess.STDOUT)
        try:
            if not wait_for_line(server_log, READY_LINE, server, SERVER_READY_TIMEOUT_S):
                stop_server(server)
                raise RuntimeError(f"server never became ready; see {server_log}")
            print("  sim:   ", " ".join(sim_cmd))
            t0 = time.monotonic()
            with open(sim_log, "w") as rlog:
                rc = subprocess.call(sim_cmd, cwd=HERE, env=env, stdout=rlog, stderr=subprocess.STDOUT)
            print(f"  sim exited {rc} after {time.monotonic() - t0:.0f} s")
        finally:
            stop_server(server)

    summary_path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(summary_path):
        return {"topomap": topomap, "trial": trial, "error": f"no summary.json (sim rc {rc}); see {sim_log}"}
    with open(summary_path) as f:
        summary = json.load(f)
    summary.update(topomap=topomap, trial=trial, sim_rc=rc)
    print(f"  reached_goal_node={summary.get('reached_goal_node')} "
          f"max_node={summary.get('max_node_reached')} steps={summary.get('steps')} "
          f"final_dist={summary.get('final_dist_to_goal')} collisions={summary.get('collisions')} "
          f"failure={summary.get('failure_mode')}")
    return summary


def analyze(sweep_dir: str, topomaps) -> dict:
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 3) if xs else None

    if not os.path.isdir(sweep_dir):
        raise SystemExit(f"no such sweep directory: {sweep_dir}")
    report = {"sweep_dir": os.path.abspath(sweep_dir), "maps": {}}
    for topomap in topomaps:
        trials = []
        for p in sorted(glob.glob(os.path.join(sweep_dir, topomap, "trial*", "summary.json")),
                        key=lambda q: int(re.search(r"trial(\d+)", q).group(1))):
            with open(p) as f:
                s = json.load(f)
            s.setdefault("trial", int(re.search(r"trial(\d+)", p).group(1)))
            trials.append({k: v for k, v in s.items() if k != "node_trace"})
        if not trials:
            continue
        reached = [bool(t.get("reached_goal_node")) for t in trials]
        fails = [t.get("failure_mode", "?") for t in trials]
        report["maps"][topomap] = {
            "n_trials": len(trials),
            "reached_goal_rate": round(sum(reached) / len(reached), 3),
            "steps_mean": _mean([t.get("steps") for t in trials]),
            "final_dist_to_goal_mean": _mean([t.get("final_dist_to_goal") for t in trials]),
            "max_node_reached_mean": _mean([t.get("max_node_reached") for t in trials]),
            "collisions_mean": _mean([t.get("collisions") for t in trials]),
            "distance_m_mean": _mean([t.get("distance_m") for t in trials]),
            "failure_modes": {m: fails.count(m) for m in sorted(set(fails))},
            "trials": trials,
        }
    with open(os.path.join(sweep_dir, "sweep_summary.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=5, help="trials per topomap")
    p.add_argument("--sweep", type=int, default=None,
                   help="sweep number -> ../sim_out/sweep<N> (default: next unused)")
    p.add_argument("--topomaps", nargs="+", default=list(STARTS),
                   help=f"topomaps to run, in order; each needs a start in STARTS ({', '.join(STARTS)})")
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--goal-radius", type=int, default=2)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--device", default=None, help="server device (default: server.py's own default)")
    p.add_argument("--python", default=sys.executable, help="interpreter for server.py and sim_robot.py")
    p.add_argument("--diffusion-policy-dir", default=DEFAULT_DIFFUSION_POLICY_DIR,
                   help="prepended to PYTHONPATH for both processes ('' to skip)")
    p.add_argument("--server-args", default="", help="extra args for server.py, as one quoted string")
    p.add_argument("--sim-args", default="", help="extra args for sim_robot.py, as one quoted string")
    p.add_argument("--analyze-only", action="store_true", help="re-aggregate an existing sweep (needs --sweep)")
    args = p.parse_args()
    args.server_args = args.server_args.split()
    args.sim_args = args.sim_args.split()

    unknown = [m for m in args.topomaps if m not in STARTS]
    if unknown:
        raise SystemExit(f"no start pose for {unknown}; add it to STARTS in run_sweep.py")
    if args.analyze_only and args.sweep is None:
        raise SystemExit("--analyze-only needs --sweep")

    sweep_num = args.sweep if args.sweep is not None else next_sweep_number()
    sweep_dir = os.path.join(SIM_OUT, f"sweep{sweep_num}")
    if args.analyze_only:
        print(json.dumps(analyze(sweep_dir, args.topomaps), indent=1))
        return

    os.makedirs(sweep_dir, exist_ok=True)
    with open(os.path.join(sweep_dir, "sweep_config.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=1)
    print(f"sweep {sweep_num}: {args.trials} trials x {args.topomaps} -> {sweep_dir}")

    results = []
    for topomap in args.topomaps:
        for trial in range(args.trials):
            out_dir = os.path.join(sweep_dir, topomap, f"trial{trial}")
            try:
                results.append(run_trial(args, topomap, trial, out_dir))
            except RuntimeError as e:
                print(f"  TRIAL FAILED: {e}")
                results.append({"topomap": topomap, "trial": trial, "error": str(e)})

    report = analyze(sweep_dir, args.topomaps)
    print("\n=== SWEEP REPORT ===")
    for m, r in report["maps"].items():
        print(f"{m}: reached {r['reached_goal_rate']:.0%} of {r['n_trials']}, "
              f"steps {r['steps_mean']}, final dist {r['final_dist_to_goal_mean']}, "
              f"max node {r['max_node_reached_mean']}, failures {r['failure_modes']}")
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"{len(errors)} trial(s) failed:")
        for e in errors:
            print(f"  {e['topomap']} trial {e['trial']}: {e['error']}")


if __name__ == "__main__":
    main()
