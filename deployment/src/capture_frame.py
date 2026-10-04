"""Capture frame(s) from a USB camera and save them to deployment/test_imgs/.

Run on the Pi:
    python capture_frame.py                  # one frame from /dev/video0
    python capture_frame.py --count 5 --interval 1.0
"""
import argparse
import os
import sys
import time
from datetime import datetime

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, "..", "test_imgs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0, help="video device index (0 = /dev/video0)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--count", type=int, default=1, help="number of frames to save")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between saved frames")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--warmup", type=int, default=10, help="frames to discard so exposure settles")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}. Check `ls /dev/video*` and `v4l2-ctl --list-devices`.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera} opened at {w}x{h}")

    for _ in range(args.warmup):
        cap.read()

    for i in range(args.count):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            sys.exit("Failed to read a frame from the camera")
        name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
        path = os.path.abspath(os.path.join(args.out_dir, name))
        cv2.imwrite(path, frame)
        print(f"Saved {path}")
        if i < args.count - 1:
            time.sleep(args.interval)

    cap.release()


if __name__ == "__main__":
    main()
