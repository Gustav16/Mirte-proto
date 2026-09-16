# Denne fil skal køres direkte på Mirte/Raspberry Pi, ikke i WSL på PC'en.
# altså der skal oprettes forbindels vires Ros2 også skal filen køres derfra.

import cv2
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


class LiveCompressedCamera(Node):
    """
    ROS2 camera subscriber for Mirte's compressed camera stream.

    This version deliberately does NOT wait for /camera/camera_info.
    It only needs /camera/image_raw/compressed, because focal-length
    calibration only requires the image itself.
    """

    def __init__(self):
        super().__init__("camera_calibration_v3")

        self.bridge = CvBridge()
        self.latest_image = None
        self.frame_count = 0

        self.subscription = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self._image_callback,
            qos_profile_sensor_data,
        )

        print("V3: subscriber oprettet")
        print("V3: venter på /camera/image_raw/compressed ...")

    def _image_callback(self, msg):
        try:
            self.latest_image = self.bridge.compressed_imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
            self.frame_count += 1

            if self.frame_count == 1:
                print("V3: første kamerabillede modtaget")

            if self.frame_count % 100 == 0:
                print(f"V3: modtaget {self.frame_count} frames")

        except Exception as e:
            self.get_logger().error(f"Kunne ikke konvertere billede: {e}")


def main():
    rclpy.init()

    camera = LiveCompressedCamera()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    image_count = 0
    window_name = "Mirte V3 - compressed ROS - SPACE = gem, q = stop"

    try:
        while rclpy.ok():

            # Behandl nye ROS-beskeder.
            # Timeout gør at programmet ikke kan hænge her for evigt.
            rclpy.spin_once(camera, timeout_sec=0.02)

            img = camera.latest_image

            if img is None:
                continue

            cv2.imshow(window_name, img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                filename = os.path.join(
                    save_dir,
                    f"image_v3_{image_count:04d}.png",
                )

                if cv2.imwrite(filename, img):
                    print(f"V3: billede {image_count} gemt: {filename}")
                    image_count += 1
                else:
                    print(f"V3: FEJL - kunne ikke gemme {filename}")

            elif key == ord("q"):
                print("V3: stopper.")
                break

    except KeyboardInterrupt:
        print("\nV3: stoppet med Ctrl+C.")

    finally:
        cv2.destroyAllWindows()
        camera.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
