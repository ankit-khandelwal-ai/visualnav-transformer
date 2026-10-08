#!/usr/bin/env python3
"""Live top-down view of a ROS 2 LaserScan topic.

Usage (on a machine with ROS 2 sourced, e.g. the robot or a laptop on the same ROS_DOMAIN_ID):
    python3 visualize_lidar_live.py --topic /scan --range 6
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanListener(Node):
    def __init__(self, topic):
        super().__init__("lidar_live_viz")
        self.xy = np.empty((0, 2))
        self.frame_id = ""
        self.count = 0
        # Lidar drivers typically publish BEST_EFFORT; sensor_data QoS matches both.
        self.create_subscription(LaserScan, topic, self.cb, qos_profile_sensor_data)

    def cb(self, msg: LaserScan):
        r = np.asarray(msg.ranges, dtype=np.float32)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r >= max(msg.range_min, 1e-3)) & (r <= msg.range_max)
        # x forward, y left (ROS convention); plotted with x up, y left
        self.xy = np.stack([r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])], axis=1)
        self.frame_id = msg.header.frame_id
        self.count += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/scan")
    ap.add_argument("--range", type=float, default=6.0, help="plot half-extent in meters")
    args = ap.parse_args()

    rclpy.init()
    node = ScanListener(args.topic)

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter([], [], s=4, c="tab:blue")
    ax.plot(0, 0, "r^", markersize=10)  # sensor
    ax.set_xlim(args.range, -args.range)  # +y (left) on the left side of the plot
    ax.set_ylim(-args.range, args.range)
    # ROS: x forward, y left -> plot y horizontally (inverted), x vertically
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("y (m, left)")
    ax.set_ylabel("x (m, forward)")
    ax.set_ylim(-args.range, args.range)
    title = ax.set_title(f"waiting for {args.topic} ...")
    plt.ion()
    plt.show()

    try:
        while plt.fignum_exists(fig.number):
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.count:
                sc.set_offsets(node.xy[:, ::-1])  # (y, x)
                title.set_text(
                    f"{args.topic} [{node.frame_id}]  {len(node.xy)} pts  scans={node.count}"
                )
            fig.canvas.draw_idle()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
