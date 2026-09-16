# Denne fil skal køres direkte på Mirte/Raspberry Pi, ikke i WSL på PC'en.
# altså der skal oprettes forbindels vires Ros2 også skal filen køres derfra.

import cv2
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class LiveRawCamera(Node):
    """
    ROS2 camera subscriber for Mirte's raw camera stream.

    This version tests /camera/image_raw instead of the compressed stream.
    It uses ROS2's sensor-data QoS and does not wait for camera_info.
    """

    def __init__(self):
        super().__init__("camera_calibration_v4")

        self.bridge = CvBridge()
        self.latest_image = None
        self.frame_count = 0

        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_callback,
            qos_profile_sensor_data,
        )

        print("V4: subscriber oprettet")
        print("V4: venter på /camera/image_raw ...")

    def _image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
            self.frame_count += 1

            if self.frame_count == 1:
                print("V4: første RAW-kamerabillede modtaget")

            if self.frame_count % 100 == 0:
                print(f"V4: modtaget {self.frame_count} frames")

        except Exception as e:
            self.get_logger().error(f"Kunne ikke konvertere RAW-billede: {e}")


def main():
    rclpy.init()

    camera = LiveRawCamera()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    image_count = 0
    window_name = "Mirte V4 - RAW ROS - SPACE = gem, q = stop"

    try:
        while rclpy.ok():

            rclpy.spin_once(camera, timeout_sec=0.02)

            img = camera.latest_image

            if img is None:
                continue

            cv2.imshow(window_name, img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                filename = os.path.join(
                    save_dir,
                    f"image_v4_{image_count:04d}.png",
                )

                if cv2.imwrite(filename, img):
                    print(f"V4: billede {image_count} gemt: {filename}")
                    image_count += 1
                else:
                    print(f"V4: FEJL - kunne ikke gemme {filename}")

            elif key == ord("q"):
                print("V4: stopper.")
                break

    except KeyboardInterrupt:
        print("\nV4: stoppet med Ctrl+C.")

    finally:
        cv2.destroyAllWindows()
        camera.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
