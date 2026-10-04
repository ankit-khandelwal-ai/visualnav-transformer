"""Check that the robot drives forward, backward, turns left and turns right through RobotControl.

Run on the Pi (wheels off the ground first, then again on the floor in a clear space), battery on:
    python drive_test.py
    python drive_test.py --v 0.1 --w 0.4 --duration 2
"""
import argparse
import atexit
import signal
import sys
import time

from robot import MAX_V, MAX_W, WHEEL_MAP, RobotControl, open_board

# name, sign of v, sign of w, what the robot should do
STEPS = [
    ("forward", +1, 0, "all four wheels roll forward; robot moves forward"),
    ("backward", -1, 0, "all four wheels roll backward; robot moves backward"),
    ("turn left", 0, +1, "left wheels roll backward, right wheels roll forward; robot rotates counter-clockwise (seen from above)"),
    ("turn right", 0, -1, "left wheels roll forward, right wheels roll backward; robot rotates clockwise (seen from above)"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v", type=float, default=0.1, help=f"speed for forward/backward in m/s (max {MAX_V})")
    p.add_argument("--w", type=float, default=MAX_W, help=f"yaw rate for turns in rad/s (max {MAX_W})")
    p.add_argument("--duration", type=float, default=2.0, help="seconds per step")
    p.add_argument("--serial-port", default=None)
    args = p.parse_args()

    if not (0 < args.v <= MAX_V and 0 < args.w <= MAX_W):
        sys.exit(f"Refusing: need 0 < v <= {MAX_V} m/s and 0 < w <= {MAX_W} rad/s")

    robot = RobotControl(open_board(args.serial_port))
    atexit.register(robot.stop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(1))  # atexit then stops the motors
    robot.stop()

    results = {}
    for name, sv, sw, expect in STEPS:
        v, w = sv * args.v, sw * args.w
        rps = robot.vel_to_wheel_rps(v, w)
        print(f"\n=== {name.upper()}: v={v:+.2f} m/s, w={w:+.2f} rad/s for {args.duration}s ===")
        print(f"Expect: {expect}")
        print("Commands:", ", ".join(f"motor {m} ({WHEEL_MAP[m][0]}) {r:+.3f} rps" for m, r in sorted(rps.items())))
        input("Press Enter to run (Ctrl-C aborts)... ")
        robot.set_velocity(v, w)
        time.sleep(args.duration)
        robot.stop()
        results[name] = input("Did it behave as expected? [y/n] ").strip().lower().startswith("y")

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name:<11} {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
