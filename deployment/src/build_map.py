"""Build an occupancy map from a recorded lidar session with BreezySLAM (lidar-only, no odometry).

On the Mac:
    python build_map.py ~/Documents/gi_work_trial/lidar_recordings/<session> --out maps/apartment1
Writes <out>.pgm + <out>.yaml (ROS map_server layout), <out>_map.png, <out>_traj.png, <out>_traj.csv.
If the map looks mirrored try --flip; if it smears, try --skip 2 (or higher) and a smaller --hole-width-mm.
"""
import argparse
import os

import cv2
import numpy as np
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser

from lidar_io import load_session

NBINS = 360


def to_scan_mm(frame, max_range_m):
    """Resample a revolution to 360 one-degree bins (mm). Closest return per bin; 0 = no return."""
    deg = (np.rad2deg(frame.angles) % 360).astype(int) % NBINS
    scan = np.full(NBINS, np.inf)
    ok = np.isfinite(frame.ranges) & (frame.ranges <= max_range_m)
    np.minimum.at(scan, deg[ok], frame.ranges[ok] * 1000.0)
    scan[~np.isfinite(scan)] = 0
    return [int(x) for x in scan]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", required=True, help="output prefix, e.g. maps/apartment1")
    ap.add_argument("--map-size-m", type=float, default=20.0, help="side of the square map (m); start is the centre")
    ap.add_argument("--pixels", type=int, default=1000)
    ap.add_argument("--max-range-m", type=float, default=12.0)
    ap.add_argument("--hole-width-mm", type=int, default=200, help="wall thickness BreezySLAM paints")
    ap.add_argument("--skip", type=int, default=1, help="use every Nth revolution (faster, less overlap needed)")
    ap.add_argument("--flip", action="store_true", help="mirror angles (overrides the recorded flag to True)")
    ap.add_argument("--seed", type=int, default=9999)
    args = ap.parse_args()

    s = load_session(args.session, flip=True if args.flip else None)
    print(s.summary())
    frames = [f for f in s.frames if not f.flags & 1][:: args.skip]  # drop low-point-count revolutions
    dt = np.diff([f.t_mono for f in frames])
    rate = 1 / float(np.median(dt))
    laser = Laser(NBINS, rate, 360.0, int(args.max_range_m * 1000))
    slam = RMHC_SLAM(laser, args.pixels, args.map_size_m, random_seed=args.seed, hole_width_mm=args.hole_width_mm)
    mapbytes = bytearray(args.pixels * args.pixels)

    traj = []
    for i, f in enumerate(frames):
        slam.update(to_scan_mm(f, args.max_range_m))
        x, y, th = slam.getpos()  # mm, mm, degrees
        traj.append((f.t_mono, x / 1000.0, y / 1000.0, th))
        if i % 100 == 0:
            print(f"{i}/{len(frames)}  pose ({x / 1000:.2f} m, {y / 1000:.2f} m, {th:.0f} deg)")
    slam.getmap(mapbytes)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    img = np.frombuffer(mapbytes, np.uint8).reshape(args.pixels, args.pixels)  # 0 = occupied, 255 = free
    res = args.map_size_m / args.pixels  # m per pixel
    cv2.imwrite(args.out + ".pgm", img)
    cv2.imwrite(args.out + "_map.png", img)
    with open(args.out + ".yaml", "w") as fh:  # ROS map_server layout (image y axis vs ROS y not verified)
        fh.write(f"image: {os.path.basename(args.out)}.pgm\nresolution: {res}\n"
                 "origin: [0.0, 0.0, 0.0]\n"  # BreezySLAM poses are measured from the map corner; start is the centre
                 
                 "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    np.savetxt(args.out + "_traj.csv", np.array(traj), delimiter=",", header="t_mono,x_m,y_m,theta_deg", comments="")

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    px = [(int(x / res), int(y / res)) for _, x, y, _ in traj]  # BreezySLAM's mm pose is in map pixels * res
    cv2.polylines(vis, [np.array(px, np.int32)], False, (0, 0, 255), 1)
    cv2.circle(vis, px[0], 4, (0, 255, 0), -1)  # start
    cv2.imwrite(args.out + "_traj.png", vis)
    print(f"wrote {args.out}.pgm/.yaml, _map.png, _traj.png, _traj.csv  (resolution {res * 100:.1f} cm/px)")


if __name__ == "__main__":
    main()
