"""Test the wheels through RobotControl, so WHEEL_MAP and the kinematics in robot.py are what's being tested.

Run on the Pi with the wheels OFF THE GROUND and the battery connected:
    python motor_test.py                          # each wheel in turn: forward, then reverse (WHEEL_MAP signs applied)
    python motor_test.py --motor 2                # only motor 2
    python motor_test.py --v 0.1 --w 0.0          # drive at 0.1 m/s for --duration seconds
    python motor_test.py --v 0.0 --w 0.3          # yaw left at 0.3 rad/s
"""
import argparse
import atexit
import signal
import sys
import time

from robot import MAX_V, MAX_W, WHEEL_MAP, RobotControl, open_board

HARD_MAX_RPS = 2.0  # refuse anything faster than this for a bring-up test


def test_wheels(robot: RobotControl, motors, rps: float, duration: float, confirm: bool):
    for motor in motors:
        side, sign = WHEEL_MAP[motor]
        for label, direction in (("forward", +1), ("reverse", -1)):
            print(f"--- Motor {motor} ({side}, sign {sign:+d}): {label} at {rps} rps for {duration}s ---")
            robot.set_wheel_rps({motor: sign * direction * rps})
            time.sleep(duration)
            robot.stop()
            time.sleep(0.5)
        print("Did that wheel spin forward, then backward?")
        if confirm and motor != motors[-1]:
            input("Press Enter for the next motor... ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serial-port", default=None, help="serial device (default: /dev/rrc, else first ttyACM/ttyUSB)")
    p.add_argument("--rps", type=float, default=0.5, help="wheel speed for the per-wheel test")
    p.add_argument("--duration", type=float, default=1.0, help="seconds per step")
    p.add_argument("--motor", type=int, choices=sorted(WHEEL_MAP), default=None, help="test only this motor")
    p.add_argument("--v", type=float, default=None, help="drive mode: forward speed in m/s")
    p.add_argument("--w", type=float, default=None, help="drive mode: yaw rate in rad/s (+ = left)")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompts")
    args = p.parse_args()

    drive_mode = args.v is not None or args.w is not None
    v, w = args.v or 0.0, args.w or 0.0
    if drive_mode and not (0 <= v <= MAX_V and abs(w) <= MAX_W):
        sys.exit(f"Refusing: need 0 <= v <= {MAX_V} m/s and |w| <= {MAX_W} rad/s")
    if abs(args.rps) > HARD_MAX_RPS:
        sys.exit(f"--rps {args.rps} exceeds the {HARD_MAX_RPS} rps safety limit for this test")

    robot = RobotControl(open_board(args.serial_port))
    atexit.register(robot.stop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(1))  # atexit then stops the motors

    robot.board.enable_reception()
    time.sleep(1.0)
    print(f"Battery (raw board report): {robot.board.get_battery()}  (None = no report yet, not necessarily an error)")

    robot.stop()
    if not args.yes:
        input("Wheels lifted off the ground and the battery switched on? Press Enter to start (Ctrl-C aborts)... ")

    if drive_mode:
        print(f"Driving v={v} m/s, w={w} rad/s for {args.duration}s; wheel rps: {robot.vel_to_wheel_rps(v, w)}")
        robot.set_velocity(v, w)
        time.sleep(args.duration)
        robot.stop()
    else:
        test_wheels(robot, [args.motor] if args.motor else sorted(WHEEL_MAP), args.rps, args.duration, not args.yes)

    print("\nDone. Motors stopped.")


if __name__ == "__main__":
    main()
