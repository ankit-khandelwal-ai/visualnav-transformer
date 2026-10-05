"""Load a scene mesh in habitat-sim, report its navmesh, and render preview frames.

    python check_scene.py ../scenes/hallway_straight.glb --pos 0.5 0 0 --yaw -90

--pos is the agent position (snapped to the navmesh); --yaw is the heading in
degrees about +Y, where 0 faces -Z (habitat's forward) and -90 faces +X.
Writes <scene>_preview.png next to the scene plus a top-down navmesh map.
"""
import argparse
import math
import os

import numpy as np
from PIL import Image

import habitat_sim
import magnum as mn
from habitat_sim.utils import common as U


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene")
    p.add_argument("--pos", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    p.add_argument("--yaw", type=float, default=-90.0)
    p.add_argument("--agent-radius", type=float, default=0.18)
    p.add_argument("--agent-height", type=float, default=0.75)
    p.add_argument("--tag", default="", help="suffix for the preview filename (multiple views)")
    args = p.parse_args()

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = args.scene
    backend_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = args.agent_height
    agent_cfg.radius = args.agent_radius
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "rgb"
    cam.resolution = [480, 640]
    cam.position = mn.Vector3(0.0, 0.65, 0.0)
    cam.hfov = 90
    agent_cfg.sensor_specifications = [cam]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    ns = habitat_sim.nav.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius = args.agent_radius
    ns.agent_height = args.agent_height
    sim.recompute_navmesh(sim.pathfinder, ns)
    pf = sim.pathfinder
    print(f"navmesh loaded={pf.is_loaded} islands={pf.num_islands} "
          f"area={pf.navigable_area:.2f} m^2 bounds={[list(np.round(b, 2)) for b in pf.get_bounds()]}")

    pos = pf.snap_point(np.array(args.pos, dtype=np.float32))
    rot = U.quat_from_angle_axis(math.radians(args.yaw), np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(habitat_sim.AgentState(position=pos, rotation=rot))
    print(f"agent at {np.round(pos, 3).tolist()} yaw {args.yaw}")
    isl = pf.get_island(pos)
    print(f"agent island={isl} island_area={pf.island_area(isl):.2f} m^2; all islands: "
          + ", ".join(f"{i}:{pf.island_area(i):.2f}" for i in range(pf.num_islands)))

    rgb = np.asarray(sim.get_sensor_observations()["rgb"])[:, :, :3]
    stem = os.path.splitext(args.scene)[0]
    tag = f"_{args.tag}" if args.tag else ""
    Image.fromarray(rgb).save(f"{stem}_preview{tag}.png")
    top = pf.get_topdown_view(0.05, float(pos[1]))
    Image.fromarray((top * 255).astype(np.uint8)).save(f"{stem}_navmesh.png")
    print(f"mean pixel {rgb.mean():.1f}; wrote {stem}_preview{tag}.png, {stem}_navmesh.png")
    sim.close()


if __name__ == "__main__":
    main()
