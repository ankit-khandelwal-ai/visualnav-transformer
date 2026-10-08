"""Fast interactive viewer for a textured GLB (Open3D, GPU rendered).

Usage: python visualize_glb.py [path.glb]
Mouse: left-drag rotate, scroll zoom, shift+left-drag pan.
"""
import os
import sys

import numpy as np
import open3d as o3d
import trimesh


def vertex_colors(mesh):
    """Per-vertex RGB in [0,1], sampled from the base-color texture at each vertex UV."""
    vis = mesh.visual
    if vis.kind == "texture" and getattr(vis.material, "baseColorTexture", None) is not None:
        tex = np.asarray(vis.material.baseColorTexture.convert("RGB")) / 255.0
        h, w = tex.shape[:2]
        uv = vis.uv % 1.0
        x = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
        y = np.clip(((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
        return tex[y, x]
    return vis.to_color().vertex_colors[:, :3] / 255.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/10_5_2026.glb")
    scene = trimesh.load(path)
    meshes = scene.dump() if isinstance(scene, trimesh.Scene) else [scene]  # dump() bakes transforms

    combined = o3d.geometry.TriangleMesh()
    for m in meshes:
        part = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
        part.vertex_colors = o3d.utility.Vector3dVector(vertex_colors(m))
        combined += part
    print(f"{len(combined.triangles):,} faces")

    # Unlit colors look like the texture; computing normals would add shading.
    o3d.visualization.draw_geometries([combined], window_name=os.path.basename(path),
                                      width=1400, height=900, mesh_show_back_face=True)


if __name__ == "__main__":
    main()
