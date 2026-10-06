"""Click on the top-down navmesh map to get --start / --yaw values for sim_robot.py.

    python pick_start.py --scene ../scenes/apartment.glb
    python pick_start.py --scene ../scenes/apartment.glb --height 0.1 --out picks

Controls (map window):
    left click        set the start position (snapped to the navmesh)
    right click       set the heading: the start will face the clicked point
    n                 keep the current pick and start a new one
    u                 undo (drop the current pick, or the last kept one)
    s                 save picks.json + picked_map.png to --out (or cwd)
    q / Esc           quit (also saves)

Each pick is printed as a ready-to-paste `--start X Y Z --yaw DEG` line, and a
second window shows the robot camera from that pose so you can sanity-check it.

The scene is loaded with the same agent radius/height and navmesh recompute as
sim_robot.py, and the map uses the same meters-per-pixel and grid convention as
its trajectory_map.png (row from z, col from x), so positions match exactly.
"""
import argparse
import json
import math
import os

import cv2
import numpy as np

import habitat_sim
import magnum as mn
from habitat_sim.utils import common as U

from sim_robot import DEFAULT_SCENES_DIR, render_topdown, resolve_scene, world_to_grid

MPP = 0.05
WIN_MAP = "pick start (L: position, R: heading, n: next, u: undo, s: save, q: quit)"
WIN_CAM = "camera from picked pose"


def grid_to_world(row: int, col: int, bounds, height: float, mpp: float = MPP):
    """Inverse of sim_robot.world_to_grid, at the centre of the cell."""
    lower, _ = bounds
    return np.array([lower[0] + (col + 0.5) * mpp, height, lower[2] + (row + 0.5) * mpp],
                    dtype=np.float32)


def yaw_towards(start, target) -> float:
    """Heading in degrees about +Y so that `start` faces `target` (0 faces -Z, -90 faces +X)."""
    dx, dz = float(target[0] - start[0]), float(target[2] - start[2])
    return math.degrees(math.atan2(-dx, -dz))


class Picker:
    def __init__(self, sim, map_img, bounds, height, scale):
        self.sim = sim
        self.map_img = map_img
        self.bounds = bounds
        self.height = height
        self.scale = scale  # display pixels per map pixel
        self.picks = []  # kept picks: {"start": [x, y, z], "yaw": deg}
        self.cur_start = None
        self.cur_yaw = 0.0
        self.dirty = True

    # ---- geometry ----
    def to_map_px(self, x_disp, y_disp):
        return int(y_disp / self.scale), int(x_disp / self.scale)  # (row, col)

    def world_to_disp(self, pos):
        r, c = world_to_grid(pos[0], pos[2], self.bounds, MPP)
        return int((c + 0.5) * self.scale), int((r + 0.5) * self.scale)

    def snap(self, row, col):
        p = grid_to_world(row, col, self.bounds, self.height)
        s = self.sim.pathfinder.snap_point(p)
        if not np.all(np.isfinite(s)):
            return None
        return np.array(s, dtype=np.float32)

    # ---- events ----
    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            s = self.snap(*self.to_map_px(x, y))
            if s is None:
                print("not navigable near that point")
                return
            self.cur_start = s
            self.dirty = True
            self.report()
        elif event == cv2.EVENT_RBUTTONDOWN and self.cur_start is not None:
            target = grid_to_world(*self.to_map_px(x, y), self.bounds, self.height)
            self.cur_yaw = yaw_towards(self.cur_start, target)
            self.dirty = True
            self.report()

    def report(self):
        x, y, z = (round(float(v), 3) for v in self.cur_start)
        print(f"--start {x} {y} {z} --yaw {self.cur_yaw:.1f}")

    def keep(self):
        if self.cur_start is None:
            return
        self.picks.append({"start": [round(float(v), 3) for v in self.cur_start],
                           "yaw": round(self.cur_yaw, 1)})
        print(f"kept pick {len(self.picks)}")
        self.cur_start, self.cur_yaw = None, 0.0
        self.dirty = True

    def undo(self):
        if self.cur_start is not None:
            self.cur_start, self.cur_yaw = None, 0.0
        elif self.picks:
            self.picks.pop()
        self.dirty = True

    # ---- rendering ----
    def camera_view(self):
        rot = U.quat_from_angle_axis(math.radians(self.cur_yaw), np.array([0.0, 1.0, 0.0]))
        self.sim.get_agent(0).set_state(habitat_sim.AgentState(position=self.cur_start, rotation=rot),
                                        reset_sensors=True)
        rgb = self.sim.get_sensor_observations()["rgb"][..., :3]
        return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

    def draw_pose(self, img, start, yaw, color, label):
        cx, cy = self.world_to_disp(start)
        cv2.circle(img, (cx, cy), 6, color, 2)
        # heading arrow: forward is (-sin yaw, -cos yaw) in (x, z); display y grows with z
        th = math.radians(yaw)
        fx, fz = -math.sin(th), -math.cos(th)
        L = 25
        cv2.arrowedLine(img, (cx, cy), (int(cx + fx * L), int(cy + fz * L)), color, 2, tipLength=0.3)
        cv2.putText(img, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def render(self):
        h, w = self.map_img.shape[:2]
        img = cv2.resize(self.map_img, (int(w * self.scale), int(h * self.scale)),
                         interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i, p in enumerate(self.picks):
            self.draw_pose(img, p["start"], p["yaw"], (0, 160, 0), str(i + 1))
        if self.cur_start is not None:
            self.draw_pose(img, self.cur_start, self.cur_yaw, (0, 0, 255), "cur")
        return img

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        picks = self.picks + ([{"start": [round(float(v), 3) for v in self.cur_start],
                                "yaw": round(self.cur_yaw, 1)}] if self.cur_start is not None else [])
        with open(os.path.join(out_dir, "picks.json"), "w") as f:
            json.dump({"height": self.height, "picks": picks}, f, indent=1)
        cv2.imwrite(os.path.join(out_dir, "picked_map.png"), self.render())
        print(f"saved {len(picks)} picks to {out_dir}/picks.json")
        for p in picks:
            print(f"  --start {p['start'][0]} {p['start'][1]} {p['start'][2]} --yaw {p['yaw']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default=os.path.join(DEFAULT_SCENES_DIR, "skokloster-castle.glb"))
    p.add_argument("--height", type=float, default=None,
                   help="floor height (y) of the navmesh slice to show; default: a random navigable point's y")
    p.add_argument("--agent-radius", type=float, default=0.18)
    p.add_argument("--agent-height", type=float, default=0.75)
    p.add_argument("--camera-height", type=float, default=0.65)
    p.add_argument("--camera-tilt", type=float, default=0.0)
    p.add_argument("--hfov", type=int, default=90)
    p.add_argument("--scale", type=float, default=None,
                   help="display zoom (pixels per 5 cm map cell); default fits ~900 px")
    p.add_argument("--out", default=".", help="where to write picks.json and picked_map.png")
    p.add_argument("--sim-gpu", type=int, default=0)
    args = p.parse_args()

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = resolve_scene(args.scene)
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

    height = args.height
    if height is None:
        height = float(sim.pathfinder.get_random_navigable_point()[1])
        print(f"using floor height y={height:.3f} (pass --height to pick another floor)")
    map_img, bounds, _ = render_topdown(sim.pathfinder, height=height, meters_per_pixel=MPP)
    h, w = map_img.shape[:2]
    scale = args.scale or max(1.0, min(900 / h, 900 / w))
    lower, upper = bounds
    print(f"scene {backend_cfg.scene_id}: map {w}x{h} cells @ {MPP} m, "
          f"x [{lower[0]:.2f}, {upper[0]:.2f}]  z [{lower[2]:.2f}, {upper[2]:.2f}]")

    picker = Picker(sim, map_img, bounds, height, scale)
    cv2.namedWindow(WIN_MAP)
    cv2.setMouseCallback(WIN_MAP, picker.on_mouse)
    try:
        while True:
            if picker.dirty:
                cv2.imshow(WIN_MAP, picker.render())
                if picker.cur_start is not None:
                    cv2.imshow(WIN_CAM, picker.camera_view())
                picker.dirty = False
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("n"):
                picker.keep()
            elif k == ord("u"):
                picker.undo()
            elif k == ord("s"):
                picker.save(args.out)
        picker.save(args.out)
    finally:
        cv2.destroyAllWindows()
        sim.close()


if __name__ == "__main__":
    main()
