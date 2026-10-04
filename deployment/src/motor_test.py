"""Spin each wheel of the MentorPi M1 one at a time through the RRC Lite board.

Run on the Pi with the wheels OFF THE GROUND and the battery connected:
    python motor_test.py                  # each motor: +0.5 rps for 1 s, then -0.5 rps for 1 s
    python motor_test.py --motor 2        # only motor 2
    python motor_test.py --port /dev/ttyACM0 --rps 0.3
"""
import argparse
import atexit
import glob
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ros_robot_controller_sdk import Board  # noqa: E402

MOTOR_IDS = [1, 2, 3, 4]
HARD_MAX_RPS = 2.0  # refuse anything faster than this for a bring-up test


def find_port():
    if os.path.exists("/dev/rrc"):
        return "/dev/rrc"
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return candidates[0] if candidates else None


def stop_all(board):
    for _ in range(3):  # repeat in case a packet is dropped
        board.set_motor_speed([[i, 0.0] for i in MOTOR_IDS])
        time.sleep(0.05)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None, help="serial device (default: /dev/rrc, else first ttyACM/ttyUSB)")
    p.add_argument("--rps", type=float, default=0.5, help="wheel speed in revolutions/second")
    p.add_argument("--duration", type=float, default=1.0, help="seconds per direction")
    p.add_argument("--motor", type=int, choices=MOTOR_IDS, default=None, help="test only this motor (1-4)")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompts")
    args = p.parse_args()

    if abs(args.rps) > HARD_MAX_RPS:
        sys.exit(f"--rps {args.rps} exceeds the {HARD_MAX_RPS} rps safety limit for this test")

    port = args.port or find_port()
    if port is None:
        sys.exit("No serial port found. Plug in the RRC Lite board and check `ls /dev/tty*`.")
    print(f"Opening {port} at 1,000,000 baud")
    board = Board(device=port)

    atexit.register(stop_all, board)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(1))  # atexit then stops the motors

    board.enable_reception()
    time.sleep(1.0)
    print(f"Battery (raw board report): {board.get_battery()}  (None = no report yet, not necessarily an error)")

    stop_all(board)
    if not args.yes:
        input("Wheels lifted off the ground and the battery switched on? Press Enter to start (Ctrl-C aborts)... ")

    for motor in [args.motor] if args.motor else MOTOR_IDS:
        print(f"\n--- Motor {motor}: +{args.rps} rps for {args.duration}s ---")
        board.set_motor_speed([[motor, args.rps]])
        time.sleep(args.duration)
        stop_all(board)
        time.sleep(0.5)

        print(f"--- Motor {motor}: -{args.rps} rps for {args.duration}s ---")
        board.set_motor_speed([[motor, -args.rps]])
        time.sleep(args.duration)
        stop_all(board)

        print(f"Note which wheel moved (front/back, left/right) and which way +{args.rps} turned it.")
        if not args.yes and motor != (args.motor or MOTOR_IDS[-1]):
            input("Press Enter for the next motor... ")

    print("\nDone. Motors stopped.")


if __name__ == "__main__":
    main()
