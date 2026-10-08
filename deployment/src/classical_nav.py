"""Classical navigation baseline: scan-to-map localization + RRT planning + pure pursuit.

Runs on the server (Mac). The robot streams its latest lidar revolution with each frame
(robot.py --lidar-stream); this returns a (v, w) command in ActionMsg.velocity, which robot.py
already executes (same path as joystick mode), plus the pose and path for display.

Map: the .pgm/.yaml written by build_map.py (BreezySLAM). Frame conventions match build_map.py:
the map frame is the image frame (x = column * res, y = row * res, metres) and a scan point at
angle a / range r sits at pose_xy + R(theta) @ (r cos a, r sin a). Use the same --lidar-flip for
recording and for streaming, or the live scans will be mirrored relative to the map.
"""
import math
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from protocol import ActionMsg, FrameMsg

OCC_THRESH = 100  # pgm pixel < this = wall (BreezySLAM: dark = occupied, ~127 = unknown, white = free)
FREE_THRESH = 200


def load_map(prefix: str):
    """prefix = path without extension, e.g. maps/apartment3. Returns (uint8 image, resolution m/px)."""
    img = cv2.imread(prefix + ".pgm", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(prefix + ".pgm")
    res = None
    for line in open(prefix + ".yaml"):
        if line.startswith("resolution:"):
            res = float(line.split(":")[1])
    if res is None:
        raise ValueError("no resolution in " + prefix + ".yaml")
    return img, res


def read_start_pose(prefix: str):
    """First row of <prefix>_traj.csv = where the SLAM run started: (x_m, y_m, theta_deg).

    BreezySLAM numbers its scan bins from -180 deg, so its heading is 180 deg off the robot's true
    forward direction (checked against its own poses); the +180 here converts to the true heading."""
    path = prefix + "_traj.csv"
    if not os.path.exists(path):
        return None
    row = np.loadtxt(path, delimiter=",", skiprows=1, max_rows=1)
    return float(row[1]), float(row[2]), float(row[3]) + 180.0


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ClassicalNavigator:
    def __init__(self, map_prefix, start_pose, goal_xy, v_max, w_max, speed=0.5, robot_radius=0.20,
                 lookahead=0.4, goal_tol=0.15, max_scan_range=6.0, seed=0, use_cmd_prior=False):
        self.img, self.res = load_map(map_prefix)
        self.v_max, self.w_max, self.speed = v_max, w_max, speed
        self.lookahead, self.goal_tol, self.max_scan_range = lookahead, goal_tol, max_scan_range
        self.goal = np.array(goal_xy, float)
        self.rng = np.random.default_rng(seed)
        self.use_cmd_prior = use_cmd_prior  # off: commanded (v, w) badly predicts skid-steer motion (replay test)
        H, W = self.img.shape
        self.H, self.W = H, W

        # likelihood field: metres to the nearest wall pixel
        wall = (self.img < OCC_THRESH)
        self.dist_wall = cv2.distanceTransform((~wall).astype(np.uint8), cv2.DIST_L2, 5) * self.res

        # planning space: free pixels at least robot_radius from any wall AND from unknown space.
        # Eroding by the robot radius also deletes the thin 1-px free streaks BreezySLAM paints along
        # beams that never hit anything, so only real open areas survive.
        r_px = max(1, int(round(robot_radius / self.res)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
        free = cv2.erode((self.img >= FREE_THRESH).astype(np.uint8), k).astype(bool) & (self.dist_wall > robot_radius)
        self.pose = np.array([start_pose[0], start_pose[1], math.radians(start_pose[2])], float)
        # keep only the region connected to the start so the planner can't pick unreachable areas
        n, lab = cv2.connectedComponents(free.astype(np.uint8))
        sx, sy = self._px(self.pose[:2])
        if not (0 <= sx < W and 0 <= sy < H) or lab[sy, sx] == 0:
            ys, xs = np.nonzero(free)
            i = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
            print(f"start pose is not in free space; planning from the nearest free cell ({xs[i]}, {ys[i]})")
            sx, sy = int(xs[i]), int(ys[i])
        self.free = lab == lab[sy, sx]
        gx, gy = self._px(self.goal)
        if not (0 <= gx < W and 0 <= gy < H) or not self.free[gy, gx]:
            raise SystemExit(f"goal {tuple(goal_xy)} (pixel {gx},{gy}) is not in the reachable free space; "
                             "see <map>_free.png written next to the map")
        self.path = None  # (M, 2) metres
        self.path_i = 0
        self.last_t = None
        self.last_cmd = (0.0, 0.0)
        self.lost = 0
        self.last_info = ""
        self.scan_xy_map = None
        self.done = False

    # --- helpers --------------------------------------------------------------------------------
    def _px(self, xy) -> Tuple[int, int]:
        return int(round(xy[0] / self.res)), int(round(xy[1] / self.res))

    def save_debug(self, prefix):
        vis = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        vis[self.free] = (0.5 * vis[self.free] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
        cv2.imwrite(prefix + "_free.png", vis)

    # --- localization: correlative scan-to-map matching against the wall likelihood field -------
    def _score(self, cands, pts, dmax=0.5):
        """cands (K,3) poses, pts (N,2) robot-frame points -> mean wall distance per candidate (K,)."""
        c, s = np.cos(cands[:, 2])[:, None], np.sin(cands[:, 2])[:, None]
        x = cands[:, 0:1] + c * pts[None, :, 0] - s * pts[None, :, 1]
        y = cands[:, 1:2] + s * pts[None, :, 0] + c * pts[None, :, 1]
        ix, iy = np.round(x / self.res).astype(int), np.round(y / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        d = np.full(ix.shape, dmax)
        d[ok] = np.minimum(self.dist_wall[iy[ok], ix[ok]], dmax)
        return d.mean(axis=1), (d < 0.1).mean(axis=1)

    def _grid(self, center, dxy, nxy, dth, nth):
        o = np.linspace(-dxy, dxy, nxy)
        t = np.linspace(-dth, dth, nth)
        gx, gy, gt = np.meshgrid(o, o, t, indexing="ij")
        return np.stack([center[0] + gx.ravel(), center[1] + gy.ravel(), center[2] + gt.ravel()], axis=1)

    def localize(self, pred, pts):
        lost = self.lost > 0
        wide = (0.5, 11, math.radians(30), 13) if lost else (0.2, 9, math.radians(12), 9)
        best = pred
        for dxy, nxy, dth, nth in (wide, (wide[0] / 4, 9, wide[2] / 4, 9), (wide[0] / 16, 9, wide[2] / 16, 9)):
            cands = self._grid(best, dxy, nxy, dth, nth)
            cost, _ = self._score(cands, pts)
            cost = cost + 0.5 * ((cands[:, 0] - pred[0]) ** 2 + (cands[:, 1] - pred[1]) ** 2)  # stay near the prior
            best = cands[int(np.argmin(cost))]
        _, inl = self._score(best[None], pts)
        return best, float(inl[0])

    # --- planning: RRT on the inflated free mask, then shortcut smoothing ------------------------
    def _line_free(self, a, b):
        n = max(2, int(np.hypot(*(b - a)) / (0.5 * self.res)))
        t = np.linspace(0, 1, n)[:, None]
        p = a[None] * (1 - t) + b[None] * t
        ix, iy = np.round(p[:, 0] / self.res).astype(int), np.round(p[:, 1] / self.res).astype(int)
        if (ix < 0).any() or (iy < 0).any() or (ix >= self.W).any() or (iy >= self.H).any():
            return False
        return bool(self.free[iy, ix].all())

    def plan(self, start, goal, step=0.3, iters=6000, goal_bias=0.1):
        ys, xs = np.nonzero(self.free)
        sx, sy = self._px(start)
        if not (0 <= sx < self.W and 0 <= sy < self.H and self.free[sy, sx]):  # snap into free space
            i = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
            start = np.array([xs[i], ys[i]], float) * self.res
        nodes, parent = [np.array(start, float)], [-1]
        arr = np.empty((iters + 2, 2))  # preallocated: np.array(nodes) every iteration was O(n^2)
        arr[0] = nodes[0]
        for _ in range(iters):
            if self.rng.random() < goal_bias:
                q = goal
            else:
                j = self.rng.integers(len(xs))
                q = np.array([xs[j], ys[j]], float) * self.res
            i = int(np.argmin(((arr[:len(nodes)] - q) ** 2).sum(1)))
            d = q - nodes[i]
            L = float(np.hypot(*d))
            if L < 1e-6:
                continue
            new = nodes[i] + d / L * min(step, L)
            if not self._line_free(nodes[i], new):
                continue
            arr[len(nodes)] = new
            nodes.append(new)
            parent.append(i)
            if np.hypot(*(new - goal)) < step and self._line_free(new, goal):
                nodes.append(np.array(goal, float))
                parent.append(len(nodes) - 2)
                path, k = [], len(nodes) - 1
                while k != -1:
                    path.append(nodes[k])
                    k = parent[k]
                return self._smooth(path[::-1])
        return None

    def _smooth(self, path, tries=150):
        path = list(path)
        for _ in range(tries):
            if len(path) < 3:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i > 1 and self._line_free(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return np.array(path)

    # --- control: pure pursuit -------------------------------------------------------------------
    def track(self):
        """(v, w) toward a lookahead point on the path; None when the goal is reached."""
        pos = self.pose[:2]
        if np.hypot(*(pos - self.goal)) < self.goal_tol:
            return None
        d = np.hypot(*(self.path - pos).T)
        self.path_i = max(self.path_i, int(np.argmin(d[self.path_i:])) + self.path_i)
        target = self.path[-1]
        for k in range(self.path_i, len(self.path) - 1):  # walk along segments to the lookahead distance
            a, b = self.path[k], self.path[k + 1]
            if np.hypot(*(b - pos)) >= self.lookahead:
                seg = b - a
                f = np.clip(np.dot(pos - a, seg) / max(np.dot(seg, seg), 1e-9), 0, 1)
                target = a + seg * f
                while np.hypot(*(target - pos)) < self.lookahead and f < 1:
                    f = min(1.0, f + 0.05)
                    target = a + seg * f
                break
        dx, dy = target - pos
        th = self.pose[2]
        xr, yr = math.cos(th) * dx + math.sin(th) * dy, -math.sin(th) * dx + math.cos(th) * dy
        alpha = math.atan2(yr, xr)  # >0 = target on the robot's left
        vmax, wmax = self.v_max * self.speed, self.w_max * self.speed
        w = float(np.clip(1.5 * alpha, -wmax, wmax))
        v = vmax * max(0.0, 1 - abs(alpha) / (math.pi / 3))  # slow down / turn in place for big errors
        return float(v), w

    # --- one server step -------------------------------------------------------------------------
    def step(self, frame: FrameMsg) -> ActionMsg:
        if frame.scan is None or len(frame.scan) == 0:
            return ActionMsg(frame.seq, frame.t_capture, status="warmup", error="no lidar scan in frame")
        t = frame.t_capture
        dt = 0.0 if self.last_t is None else float(np.clip(t - self.last_t, 0.0, 0.5))
        self.last_t = t
        if frame.seq == 0:  # new run: forget the previous path
            self.path, self.path_i, self.lost, self.done = None, 0, 0, False

        ang, rng = frame.scan[:, 0], frame.scan[:, 1]
        ok = np.isfinite(rng) & (rng > 0.1) & (rng < self.max_scan_range)
        ang, rng = ang[ok], rng[ok]
        pts = np.stack([rng * np.cos(ang), rng * np.sin(ang)], axis=1)
        if len(pts) > 150:
            pts = pts[np.linspace(0, len(pts) - 1, 150).astype(int)]

        # predict (hold the last pose; no odometry), then correct by matching the scan to the map
        v, w = self.last_cmd if self.use_cmd_prior else (0.0, 0.0)
        th = self.pose[2] + w * dt
        pred = np.array([self.pose[0] + v * math.cos(th) * dt, self.pose[1] + v * math.sin(th) * dt, th])
        if len(pts) >= 20:
            best, inl = self.localize(pred, pts)
        else:
            best, inl = pred, 0.0
        if inl >= 0.15:  # the SLAM map is sparse, so even a perfect pose only scores ~0.4
            self.pose, self.lost = best, 0
        else:
            self.pose, self.lost = pred, self.lost + 1  # keep dead reckoning and search wider next time
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        self.scan_xy_map = self.pose[:2] + pts @ np.array([[c, s], [-s, c]])
        info = f"inliers {inl:.2f}" + (f"  LOST x{self.lost}" if self.lost else "")

        status, cmd = "ok", (0.0, 0.0)
        if self.lost > 10:
            status = "lost"
        elif self.done or np.hypot(*(self.pose[:2] - self.goal)) < self.goal_tol:
            self.done = True
            status = "goal"
        else:
            deviated = self.path is not None and np.min(np.hypot(*(self.path - self.pose[:2]).T)) > 0.5
            if self.path is None or deviated:
                t0 = time.perf_counter()
                self.path = self.plan(self.pose[:2], self.goal)
                self.path_i = 0
                info += f"  replan {1000 * (time.perf_counter() - t0):.0f} ms" + ("" if self.path is not None else " FAILED")
            if self.path is None:
                status = "noplan"
            else:
                out = self.track()
                if out is None:
                    self.done, status = True, "goal"
                else:
                    cmd = out
                    front = np.abs(ang) < math.radians(30)  # emergency stop: obstacle right in front
                    if front.any() and rng[front].min() < 0.30:
                        cmd, info = (0.0, cmd[1]), info + "  OBSTACLE"
        self.last_cmd = cmd
        self.last_info = info
        return ActionMsg(frame.seq, frame.t_capture, status=status, velocity=[cmd[0], cmd[1]],
                         pose=[float(self.pose[0]), float(self.pose[1]), float(self.pose[2])],
                         path=None if self.path is None else self.path.tolist(), error="" if status == "ok" else info,
                         info=info)

    # --- display ---------------------------------------------------------------------------------
    def render(self, frame: FrameMsg, action: ActionMsg, size=800) -> np.ndarray:
        vis = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        f = lambda xy: (int(round(xy[0] / self.res)), int(round(xy[1] / self.res)))
        if self.path is not None:
            cv2.polylines(vis, [np.array([f(p) for p in self.path], np.int32)], False, (0, 160, 0), 2)
        if self.scan_xy_map is not None:
            for p in self.scan_xy_map:
                cv2.circle(vis, f(p), 1, (0, 200, 255), -1)
        cv2.circle(vis, f(self.goal), 6, (0, 0, 255), 2)
        c = f(self.pose[:2])
        tip = f(self.pose[:2] + 0.4 * np.array([math.cos(self.pose[2]), math.sin(self.pose[2])]))
        cv2.arrowedLine(vis, c, tip, (255, 0, 0), 2, tipLength=0.4)
        # crop around the robot so the walls are big enough to read
        half = int(3.5 / self.res)
        x0, y0 = max(0, min(c[0] - half, self.W - 2 * half)), max(0, min(c[1] - half, self.H - 2 * half))
        vis = cv2.resize(vis[y0:y0 + 2 * half, x0:x0 + 2 * half], (size, size), interpolation=cv2.INTER_NEAREST)
        v, w = action.velocity or (0.0, 0.0)
        for i, text in enumerate([f"seq {frame.seq}  {action.status}", f"v {v:+.2f} m/s  w {w:+.2f} rad/s",
                                  f"pose ({self.pose[0]:.2f}, {self.pose[1]:.2f}) {math.degrees(self.pose[2]):.0f} deg",
                                  self.last_info]):
            cv2.putText(vis, text, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return vis
