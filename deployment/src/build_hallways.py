"""Build two simple hallway scene meshes for habitat-sim, with trimesh.

Generates:

    deployment/scenes/hallway_straight.glb      10 m corridor along +X, both ends capped
    deployment/scenes/hallway_left_turn.glb     +X for ~4 m, 90 deg LEFT (toward -Z), 6 m more

Geometry is authored in habitat's frame (meters, +Y up, facing +X means left
is -Z), then rotated +90 deg about X before export (to_z_up), because
habitat-sim treats a bare GLB scene as Z-up (Matterport convention) and
rotates it back to Y-up on load. Net effect: habitat coordinates == the
authoring coordinates below.
Walls/ceiling are thick boxes (>= 0.1 m) so navmesh generation treats them as
solid obstacles instead of paper-thin double-sided faces. Surfaces get muted
base colors plus a subtle procedural noise texture (numpy -> PIL, no
downloads) so floor/wall/ceiling read clearly in renders.
"""
import argparse
import os

import numpy as np
from PIL import Image

import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "scenes"))

WALL_T = 0.15      # wall thickness (m), comfortably above the 0.1 m minimum
CEIL_T = 0.15      # ceiling slab thickness
WIDTH = 2.0        # interior corridor width
HEIGHT = 2.5       # interior corridor height

# Muted palette: (base RGB 0-255, texture scale multiplier, noise sigma)
FLOOR_COLOR = (196, 198, 202, 255)   # light gray, faint checker
WALL_COLOR = (224, 216, 203, 255)    # warm off-white
CEIL_COLOR = (140, 144, 152, 255)    # darker cool gray
END_COLOR = (188, 178, 168, 255)     # slightly darker taupe for the end caps

ROOF_SLOPE_DEG = 60.0   # pitched roof top: too steep to be walkable

TEX_SIZE = 128
TEX_SCALE = 1.0     # texture repeats every TEX_SCALE meters


def noise_texture(rgb, sigma, checker=None, seed=0):
    """Small PIL image of a flat muted color plus subtle per-pixel noise.

    checker: optional (n, m) tile counts for a faint 2-tone checkerboard, so
    the floor reads as floor at a glance.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((TEX_SIZE, TEX_SIZE, 4), dtype=np.float32)
    img[..., :3] = np.array(rgb[:3], dtype=np.float32)
    img[..., 3] = rgb[3]
    if checker is not None:
        ny, nx = checker
        rows = (np.arange(TEX_SIZE) * ny // TEX_SIZE) % 2
        cols = (np.arange(TEX_SIZE) * nx // TEX_SIZE) % 2
        mask = (rows[:, None] ^ cols[None, :]).astype(np.float32)
        img[..., :3] -= 14.0 * mask[..., None]
    img[..., :3] += rng.normal(0.0, sigma, (TEX_SIZE, TEX_SIZE, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGBA")


def make_material(name, rgb, sigma, checker=None, seed=0):
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorTexture=noise_texture(rgb, sigma, checker, seed),
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )


def _apply_uv(m, mat):
    """Box-projected UVs: each face uses the two axes it does not face along,
    so the texture tiles every TEX_SCALE m on every face without stretching."""
    m = m.copy()
    m.unmerge_vertices()                       # per-face vertices -> per-face UVs
    n = np.abs(m.face_normals)
    dom = n.argmax(axis=1)                     # dominant normal axis per face
    uv = np.zeros((len(m.vertices), 2))
    pairs = {0: (2, 1), 1: (0, 2), 2: (0, 1)}
    for f, d in enumerate(dom):
        a, b = pairs[d]
        idx = m.faces[f]
        uv[idx, 0] = m.vertices[idx, a]
        uv[idx, 1] = m.vertices[idx, b]
    m.visual = trimesh.visual.texture.TextureVisuals(uv=uv / TEX_SCALE, material=mat)
    return m


def add_aabb(scene, lo, hi, mat, name):
    """Solid axis-aligned box spanning [lo, hi] (Y-up authoring frame)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2.0)
    scene.add_geometry(_apply_uv(m, mat), node_name=name, geom_name=name)


def add_roof(scene, x0, x1, z0, z1, mat, name):
    """Ceiling slab from y=HEIGHT up, with a steep (ROOF_SLOPE_DEG) pitched top.

    Its underside is a flat ceiling at y=HEIGHT, but the top faces are too steep
    for recast to mark walkable, so no navmesh island forms on the roof. The
    ridge runs along the slab's longer horizontal axis.
    """
    y0 = HEIGHT
    y1 = HEIGHT + CEIL_T
    t = np.tan(np.radians(ROOF_SLOPE_DEG))
    if (x1 - x0) >= (z1 - z0):                 # ridge along X
        zc, half = (z0 + z1) / 2.0, (z1 - z0) / 2.0
        yr = y1 + half * t
        pts = [[x, y, z] for x in (x0, x1) for (y, z) in
               ((y0, z0), (y0, z1), (y1, z0), (y1, z1), (yr, zc))]
    else:                                      # ridge along Z
        xc, half = (x0 + x1) / 2.0, (x1 - x0) / 2.0
        yr = y1 + half * t
        pts = [[x, y, z] for z in (z0, z1) for (y, x) in
               ((y0, x0), (y0, x1), (y1, x0), (y1, x1), (yr, xc))]
    m = trimesh.convex.convex_hull(np.array(pts))
    scene.add_geometry(_apply_uv(m, mat), node_name=name, geom_name=name)


def materials():
    return dict(
        floor=make_material("floor_mat", FLOOR_COLOR, 5.0, checker=(4, 4), seed=1),
        wall=make_material("wall_mat", WALL_COLOR, 4.0, seed=2),
        ceil=make_material("ceil_mat", CEIL_COLOR, 4.0, seed=3),
        end=make_material("end_mat", END_COLOR, 4.0, seed=4),
    )


def build_straight(length=10.0):
    """Straight corridor along +X, interior x in [0, length], z in [-1, 1].

    Both ends are capped (entrance wall at x=0, dead end at x=length), so the
    corridor is fully enclosed. Start near x=0.5 facing +X (habitat yaw -90).
    """
    T, w, H = WALL_T, WIDTH / 2.0, HEIGHT
    M = materials()
    s = trimesh.Scene()
    add_aabb(s, [-T, -T, -w - T], [length + T, 0, w + T], M["floor"], "floor")
    add_roof(s, -T, length + T, -w - T, w + T, M["ceil"], "ceiling")
    add_aabb(s, [-T, 0, -w - T], [length + T, H, -w], M["wall"], "wall_left")    # -Z = left facing +X
    add_aabb(s, [-T, 0, w], [length + T, H, w + T], M["wall"], "wall_right")     # +Z = right facing +X
    add_aabb(s, [-T, 0, -w], [0, H, w], M["end"], "entrance_cap")
    add_aabb(s, [length, 0, -w], [length + T, H, w], M["end"], "end_cap")
    return s


def build_left_turn(straight=4.0, after=6.0):
    """Corridor along +X, then a 90 degree LEFT turn, then a dead end.

    Habitat frame (Y up): facing +X, the agent's left is -Z. So:
      leg 1: interior x in [0, straight + 1], z in [-1, 1]   (walk +X)
      corner square: x in [straight - 1, straight + 1], z in [-1, 1]
      leg 2: interior x in [straight - 1, straight + 1],
             z in [-1 - after, -1]                           (walk -Z)
    i.e. leg 2's centerline is at x=straight, and it continues `after` m past
    the corner square. Floor/ceiling are L-shaped (one slab per leg) and cover
    only the interior plus the wall footprints. Entrance (x=0) and leg-2 end
    are capped. Start near x=0.5 facing +X (habitat yaw -90).
    """
    T, w, H = WALL_T, WIDTH / 2.0, HEIGHT
    S = straight
    xo = S + w                     # outer (far) wall inner face, x
    xi = S - w                     # leg-2 inner wall inner face, x
    ze = -w - after                # leg-2 dead-end inner face, z
    M = materials()
    s = trimesh.Scene()
    # L-shaped floor + ceiling: leg 1 slab (incl. corner) and leg 2 slab
    add_aabb(s, [-T, -T, -w - T], [xo + T, 0, w + T], M["floor"], "floor_leg1")
    add_aabb(s, [xi - T, -T, ze - T], [xo + T, 0, -w - T], M["floor"], "floor_leg2")
    add_roof(s, -T, xo + T, -w - T, w + T, M["ceil"], "ceiling_leg1")
    add_roof(s, xi - T, xo + T, ze - T, -w - T, M["ceil"], "ceiling_leg2")
    # entrance cap (x=0)
    add_aabb(s, [-T, 0, -w], [0, H, w], M["end"], "entrance_cap")
    # leg 1 right/outer wall (z=+1), runs the full length to the far corner
    add_aabb(s, [-T, 0, w], [xo + T, H, w + T], M["wall"], "wall_leg1_right")
    # far wall (x=S+1): closes leg 1 and continues as leg 2's right/outer wall
    add_aabb(s, [xo, 0, ze - T], [xo + T, H, w], M["wall"], "wall_far_outer")
    # leg 1 left/inner wall (z=-1), stops where leg 2 opens (x=S-1)
    add_aabb(s, [-T, 0, -w - T], [xi, H, -w], M["wall"], "wall_leg1_left")
    # leg 2 left/inner wall (x=S-1), from the corner to the dead end
    add_aabb(s, [xi - T, 0, ze - T], [xi, H, -w - T], M["wall"], "wall_leg2_left")
    # leg 2 dead end
    add_aabb(s, [xi, 0, ze - T], [xo, H, ze], M["end"], "end_cap")
    return s


def to_z_up(scene):
    """Rotate the Y-up authored scene so +Y -> +Z (and +Z -> -Y).

    habitat-sim treats a bare GLB scene as Z-up and rotates it -90 deg about X
    on load; pre-rotating +90 deg about X makes the two cancel, so the
    habitat frame equals the authoring frame above.
    """
    R = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    scene.apply_transform(R)
    return scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=OUT_DIR, help="where to write the .glb files")
    p.add_argument("--length", type=float, default=10.0,
                   help="straight hallway length (m)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    jobs = [
        ("hallway_straight.glb", to_z_up(build_straight(args.length))),
        ("hallway_left_turn.glb", to_z_up(build_left_turn())),
    ]
    for fname, scene in jobs:
        path = os.path.join(args.out_dir, fname)
        scene.export(path)
        watertight = all(g.is_watertight for g in scene.geometry.values())
        print(f"{path}: {len(scene.geometry)} meshes, watertight={watertight}")


if __name__ == "__main__":
    main()