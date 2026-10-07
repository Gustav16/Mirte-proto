"""MIRTE camera frames with the time they were taken, for use instead of KU_Mirte.get_image_compressed().

KU_Mirte's camera subscriber uses reliable delivery and drops the frame's timestamp. Over a weak network
(a phone hotspot) a lost packet is resent, so a frame taken before or during a move can arrive seconds later,
and there is no way to tell. This subscriber
  - uses best-effort delivery: a frame with a lost packet is dropped instead of delaying the newer ones, and
  - keeps the capture time of each frame, in the laptop's clock, so frames taken before a move can be skipped.
"""

import threading
import time

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class FreshCamera(Node):
    def __init__(self, topic='/camera/image_raw/compressed'):
        super().__init__('fresh_camera')
        self._lock = threading.Lock()
        self._image = None
        self._capture_time = None
        # laptop time - robot timestamp, the smallest seen = clock difference + the smallest network delay
        self._clock_offset = None
        self.has_timestamps = None # None until the first frame, then True/False
        self.create_subscription(CompressedImage, topic, self._callback, qos_profile_sensor_data)

    def _callback(self, msg):
        arrival = time.time()
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.has_timestamps = stamp > 0
        if self.has_timestamps:
            offset = arrival - stamp
            if self._clock_offset is None or offset < self._clock_offset:
                self._clock_offset = offset
            capture_time = stamp + self._clock_offset
        else:
            capture_time = arrival # no timestamp from the robot: the arrival time is the best we have
        with self._lock:
            self._image = img
            self._capture_time = capture_time

    def latest(self):
        """(image, capture time in the laptop's time.time() clock) of the newest frame, or (None, None)."""
        with self._lock:
            return self._image, self._capture_time
