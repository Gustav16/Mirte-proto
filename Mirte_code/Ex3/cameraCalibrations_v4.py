import cv2
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


TARGET_IMAGES = 10


class StillCompressedCamera(Node):
    """
    V4 based on the working V3.

    Uses the compressed ROS2 camera stream:
        /camera/image_raw/compressed

    Behaviour:
    - The first received frame is shown as a still image.
    - The subscriber keeps receiving new frames in the background.
    - SPACE replaces the displayed still image with the newest frame
      and saves that same frame to disk.
    - q or Ctrl+C stops the program cleanly.
    """

    def __init__(self):
        super().__init__("camera_calibration_v4")

        self.bridge = CvBridge()
        self.latest_image = None
        self.first_frame_received = False
        self.frame_count = 0

        self.subscription = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self._image_callback,
            qos_profile_sensor_data,
        )

        print("V4: subscriber oprettet")
        print("V4: venter på /camera/image_raw/compressed ...")

    def _image_callback(self, msg):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            self.latest_image = image
            self.frame_count += 1

            if not self.first_frame_received:
                self.first_frame_received = True
                print("V4: første kamerabillede modtaget")

        except Exception as e:
            self.get_logger().error(f"Kunne ikke konvertere billede: {e}")


def main():
    rclpy.init()

    camera = StillCompressedCamera()

    save_dir = os.path.dirname(os.path.abspath(__file__))
    image_count = 0

    displayed_image = None

    window_name = (
        "Mirte V4 - SPACE = nyt stillbillede + gem | q = stop"
    )

    try:
        while rclpy.ok():

            # Keep receiving camera frames from Mirte.
            rclpy.spin_once(camera, timeout_sec=0.02)

            # Show the first frame once, then keep it frozen.
            if displayed_image is None and camera.latest_image is not None:
                displayed_image = camera.latest_image.copy()
                print("V4: stillbillede klar.")
                print(
                    "V4: Tryk SPACE for at hente det nyeste billede og gemme det."
                )

            if displayed_image is None:
                continue

            # Important: we show the frozen image, not camera.latest_image.
            cv2.imshow(window_name, displayed_image)

            key = cv2.waitKey(10) & 0xFF

            if key == ord(" "):
                if camera.latest_image is None:
                    print("V4: intet kamerabillede tilgængeligt endnu.")
                    continue

                # Take a snapshot of the newest frame.
                displayed_image = camera.latest_image.copy()

                filename = os.path.join(
                    save_dir,
                    f"image_v4_{image_count:04d}.png",
                )

                if cv2.imwrite(filename, displayed_image):
                    image_count += 1
                    print(
                        f"V4: billede {image_count}/{TARGET_IMAGES} gemt: "
                        f"{filename}"
                    )

                    if image_count == TARGET_IMAGES:
                        print(
                            "V4: 10 billeder er nu gemt. "
                            "Du kan trykke q for at afslutte."
                        )
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
