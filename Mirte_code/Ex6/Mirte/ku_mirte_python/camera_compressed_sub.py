import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraCompressedSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        # Create a subscriber to the 'camera/image_raw' topic
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self._listener_callback,
            1  # Queue size
        )
        self.subscription  # prevent unused variable warning
        
        # Create a one-time subscriber to the 'camera/camera_info' topic
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self._camera_info_callback,
            10  # Queue size
        )
        self.camera_info_subscription  # prevent unused variable warning
        
        # Initialize variables to store K and P matrices
        self.k_matrix = None # intrinsic matrix 
        self.d_matrix = None # distortion matrix
        self.p_matrix = None # projection matrix

        # Initialize CvBridge to convert ROS image messages to OpenCV
        self.bridge = CvBridge()

        # Initialize a variable to store the latest image
        self.latest_image = None

        # Spin until camera info is received
        while self.k_matrix is None:
            rclpy.spin_once(self) 
        
        # Destroy the subscription after receiving the data
        self.destroy_subscription(self.camera_info_subscription)

        # Spin until the first image is received
        while self.latest_image is None:
            rclpy.spin_once(self)


    def _listener_callback(self, msg):
        """Callback function to process incoming camera images."""
        try:
            # Convert the ROS Image message to a CV2 image (BGR format)
            self.latest_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def _camera_info_callback(self, msg):
        """Callback function to process incoming camera info."""
        #self.get_logger().info("Received camera info")
        try:
            # Extract K and P matrices from the CameraInfo message
            self.k_matrix = np.array(msg.k).reshape((3, 3))
            self.d_matrix = np.array(msg.d)
            self.p_matrix = np.array(msg.p)

            # Destroy the subscription after receiving the data
        except Exception as e:
            self.get_logger().error(f"Failed to process camera info: {e}")

    def image(self):
        """Returns the most recent image received."""
        return self.latest_image

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraCompressedSubscriber()

    # Spin the node in a separate thread so we can update the image continuously
    rclpy.spin_once(camera_subscriber)

    # Loop to display the latest image
    while rclpy.ok():
        # Check if a new image is available
        img = camera_subscriber.image()

        if img is not None:
            # Display the image using OpenCV
            cv2.imshow("Camera Image", img)
        
        # Wait for 1 millisecond to update the OpenCV window
        key = cv2.waitKey(1)

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

        # Spin the node to allow ROS2 callbacks to be processed
        rclpy.spin_once(camera_subscriber)

    # Clean up
    camera_subscriber.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
