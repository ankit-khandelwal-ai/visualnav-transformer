"""Exploration metrics shared by sim_robot.py and run_explore_sweep.py.

Everything here derives from three per-run artifacts: the agent's positions
(1 per step, world frame), the per-step collision flags, and the coverage
curve (coverage fraction after each step, from CoverageTracker).

Metrics:
  - coverage_curve: coverage vs. step; written as CSV for plotting.
  - steps_per_new_cell: steps spent per unit of new area discovered (rising
    curve = revisiting/dithering instead of exploring).
  - revisit_ratio: distance traveled per unit of *new* coverage gained.
  - path_efficiency: pathfinder geodesic start->goal / actual path length
    (only meaningful when the goal is fixed; exploration runs have no goal).
  - heading_dither: residual heading after removing the smooth (median-
    filtered) component — high values = turn jitter, not committed turns.
  - failure_mode: one of timeout | stuck | pinned | dithering | overshoot |
    ok. Classification is conservative: pinned/stuck both mean "stopped
    moving", differentiated by whether collisions were happening at the end.
"""
import json
import os
from typing import List, Optional, Tuple

import numpy as np

import habitat_sim


def geodesic_distance(pathfinder, start: np.ndarray, goal: np.ndarray) -> Optional[float]:
    """Pathfinder shortest-path length, or None when start/goal aren't connected."""
    sp = habitat_sim.MultiGoalShortestPath()
    sp.requested_start = np.asarray(start, dtype=np.float32)
    sp.requested_ends = [np.asarray(goal, dtype=np.float32)]
    if not pathfinder.find_path(sp):
        return None
    return float(sp.geodesic_distance)


def steps_per_new_cell(curve: List[Optional[float]]) -> Optional[float]:
    """Steps spent per unit of new coverage gained, over the whole run.

    0 is perfect (every step finds new area); a rising curve pushes this up.
    None if no new coverage was ever gained (agent never really moved).
    """
    gained = [c for c in curve if c is not None]
    if not gained:
        return None
    new_area = gained[-1] - gained[0]
    if new_area <= 1e-6:
        return None
    return len(curve) / new_area


def revisit_ratio(curve: List[Optional[float]], distance_m: float) -> Optional[float]:
    """Meters traveled per unit of new coverage gained. Lower = more efficient."""
    gained = [c for c in curve if c is not None]
    if not gained or distance_m <= 0:
        return None
    new_area = gained[-1] - gained[0]
    if new_area <= 1e-6:
        return None
    return distance_m / new_area


def heading_dither(headings_rad: np.ndarray, window: int = 5) -> Optional[float]:
    """Std of the heading *derivative* residuals around a median-filtered trend.

    Median filter captures committed turning (the smooth trend); what's left is
    dithering. Values are in radians/step; >0.3 rad/step (~17 deg/step at 4 Hz)
    means the agent is jittering rather than steering.
    """
    if len(headings_rad) < window + 2:
        return None
    d = np.diff(np.unwrap(headings_rad))
    # median filter of width `window` via sliding median
    pad = window // 2
    padded = np.pad(d, pad, mode="edge")
    trend = np.array([np.median(padded[i:i + window]) for i in range(len(d))])
    return float(np.std(d - trend))


def classify_failure(positions: np.ndarray, collision_flags: List[bool],
                     stuck_terminated: bool, coverage_curve: List[Optional[float]],
                     max_steps: int, goal: Optional[np.ndarray],
                     geodesic: Optional[float]) -> str:
    """One of: ok | timeout | stuck | pinned | dithering | overshoot."""
    n = len(positions)
    tail = positions[-min(n, 10):]
    tail_move = float(np.linalg.norm(np.diff(tail, axis=0), axis=1).sum()) if n > 1 else 0.0
    tail_colliding = bool(np.any(collision_flags[-min(n, 10):])) if collision_flags else False
    gained = [c for c in coverage_curve if c is not None]
    # coverage gained over the whole run vs. the last 40% of steps
    late_gain = (gained[-1] - gained[int(len(gained) * 0.6)]) if len(gained) > 5 else 0.0
    early_gain = (gained[int(len(gained) * 0.6)] - gained[0]) if len(gained) > 5 else 0.0

    if stuck_terminated:
        return "pinned" if tail_colliding else "stuck"
    if n < max_steps:
        # terminated for some other reason before the budget ran out
        return "early_exit"
    if early_gain > 1e-6 and late_gain <= 1e-6 and tail_move > 0.3:
        # still moving but stopped discovering new area: circling / revisiting
        return "dithering"
    if goal is not None and geodesic is not None and geodesic > 0:
        final_d = float(np.linalg.norm(positions[-1] - np.asarray(goal)))
        start_d = float(np.linalg.norm(positions[0] - np.asarray(goal)))
        if final_d > start_d + 1.0:
            return "overshoot"
    return "timeout"


def write_coverage_csv(out_dir: str, curve: List[Optional[float]]) -> None:
    with open(os.path.join(out_dir, "coverage_curve.csv"), "w") as f:
        f.write("step,coverage\n")
        for i, c in enumerate(curve):
            f.write(f"{i},{'' if c is None else round(c, 5)}\n")


def augment_summary(summary: dict, positions: np.ndarray, collision_flags: List[bool],
                    coverage_curve: List[Optional[float]], max_steps: int,
                    goal: Optional[np.ndarray], pathfinder) -> dict:
    """Add the exploration metrics to a run's summary dict (mutates and returns)."""
    geodesic = None
    if goal is not None and pathfinder is not None:
        g = geodesic_distance(pathfinder, positions[0], np.asarray(goal))
        if g is not None and g > 0:
            geodesic = g
    distance_m = summary.get("distance_m")
    headings = np.arctan2(np.diff(positions[:, 2]), np.diff(positions[:, 0]))
    summary["path_efficiency"] = (
        round(min(geodesic / distance_m, 1.0), 3)
        if geodesic and distance_m else None)
    spnc = steps_per_new_cell(coverage_curve)
    rr = revisit_ratio(coverage_curve, distance_m)
    summary["steps_per_new_cell"] = None if spnc is None else round(spnc, 1)
    summary["revisit_ratio"] = None if rr is None else round(rr, 1)
    hd = heading_dither(headings)
    summary["heading_dither_rad_per_step"] = None if hd is None else round(hd, 3)
    summary["failure_mode"] = classify_failure(
        positions, collision_flags,
        summary.get("terminated_stuck", False),
        coverage_curve, max_steps, goal, geodesic)
    return summary