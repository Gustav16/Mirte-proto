#!/usr/bin/env python3
"""Grab a camera frame from the Mirte and open it on the Mac. Never moves the robot.

Runs on the MAC, over plain ROS 2 -- no SSH, no tunnel, no rosbridge. Requires
the Mac to be on a network without client isolation (so not the iPhone hotspot)
and the ros-humble env, which matches the robot's distro:

    conda activate ros-humble
    ./mac_grab_image.py                 # save frame.jpg and open it in Preview
    ./mac_grab_image.py -n 5            # five frames into ./frames/
    ./mac_grab_image.py --raw           # uncompressed topic instead
    ./mac_grab_image.py --no-open       # just write the file

If it hangs on "waiting for a frame", the camera is not reaching this machine --
check `ros2 topic hz /camera/image_raw/compressed` first, and remember that a
shrunken `ros2 topic list` is usually a stale CLI daemon, not the network
(`ros2 topic list --no-daemon` to rule that out).
"""

import argparse
import os
import subprocess
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

COMPRESSED_TOPIC = "/camera/image_raw/compressed"
RAW_TOPIC = "/camera/image_raw"


class Grabber(Node):
    def __init__(self, topic, raw, wanted):
        super().__init__("mac_grab_image")
        self.bridge = CvBridge()
        self.frames = []
        self.wanted = wanted
        # qos_profile_sensor_data is BEST_EFFORT. Camera drivers publish that way,
        # and a default (RELIABLE) subscriber will never match them -- you get a
        # subscription that connects to nothing and waits forever, with no error.
        # This is the single most common reason a ROS 2 image subscriber "hangs".
        self.create_subscription(
            Image if raw else CompressedImage, topic, self.on_image, qos_profile_sensor_data
        )

    def on_image(self, msg):
        if len(self.frames) >= self.wanted:
            return
        if isinstance(msg, CompressedImage):
            img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        else:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.frames.append(img)
        self.get_logger().info(f"got frame {len(self.frames)}/{self.wanted}  {img.shape}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", "--count", type=int, default=1, help="number of frames (default 1)")
    ap.add_argument("--raw", action="store_true", help=f"use {RAW_TOPIC} instead of the compressed topic")
    ap.add_argument("--topic", help="override the topic name entirely")
    ap.add_argument("--timeout", type=float, default=20.0, help="seconds to wait (default 20)")
    ap.add_argument("--no-open", action="store_true", help="do not open the image afterwards")
    args = ap.parse_args()

    topic = args.topic or (RAW_TOPIC if args.raw else COMPRESSED_TOPIC)

    rclpy.init()
    node = Grabber(topic, args.raw, args.count)
    print(f"listening on {topic} ...")

    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    while len(node.frames) < args.count:
        if node.get_clock().now().nanoseconds > deadline:
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(
                f"timed out after {args.timeout:g}s with {len(node.frames)} frame(s).\n"
                f"  ros2 topic hz {topic}          <- is it publishing at all?\n"
                f"  ros2 topic list --no-daemon    <- rule out a stale CLI daemon"
            )
        rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()

    outdir = "." if args.count == 1 else "frames"
    os.makedirs(outdir, exist_ok=True)
    written = []
    for i, img in enumerate(node.frames):
        path = os.path.join(outdir, "frame.jpg" if args.count == 1 else f"frame_{i:04d}.jpg")
        cv2.imwrite(path, img)
        written.append(os.path.abspath(path))
        print(f"saved {path}  {img.shape[1]}x{img.shape[0]}")

    if not args.no_open:
        # `open` hands it to Preview. Beats cv2.imshow, which needs a GUI-enabled
        # OpenCV build and leaves a window that has to be pumped by waitKey.
        subprocess.run(["open"] + written, check=False)


if __name__ == "__main__":
    main()
