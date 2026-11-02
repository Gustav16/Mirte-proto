import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry  # Updated import

class PositionSubscriber(Node):
    def __init__(self):
        super().__init__('odom_listener_node')

        self.latest_transform = None
        self.latest_rotation = None

        # Create a subscription to the /odom topic
        self.tf_subscription = self.create_subscription(
            Odometry,  # Updated message type
            '/odom',
            self.tf_callback,
            10  # Queue size
        )

    def tf_callback(self, msg):
        # Extract the position and orientation from the Odometry message
        self.latest_transform = msg.pose.pose.position
        self.latest_rotation = msg.pose.pose.orientation

        # Log the received data
        #self.get_logger().info(f"Position: x={self.latest_transform.x}, y={self.latest_transform.y}, z={self.latest_transform.z}")
        #self.get_logger().info(f"Rotation: x={self.latest_rotation.x}, y={self.latest_rotation.y}, z={self.latest_rotation.z}, w={self.latest_rotation.w}")

    def get_position(self):
        return self.latest_transform

    def get_rotation(self):
        return self.latest_rotation

def main():
    rclpy.init()

    # Create an instance of the PositionSubscriber node
    position_node = PositionSubscriber()

    # Loop and spin non-blocking
    while rclpy.ok():
        # Spin once to process incoming messages
        print("Spinning")
        rclpy.spin_once(position_node, timeout_sec=None)  # Adjust the timeout to control the loop rate

        # Retrieve and process position and rotation
        transform = position_node.get_position()
        rotation = position_node.get_rotation()

        if transform is not None and rotation is not None:
            print(f"Position: x={transform.x}, y={transform.y}, z={transform.z}")
            print(f"Rotation: x={rotation.x}, y={rotation.y}, z={rotation.z}, w={rotation.w}")
        else:
            print("Waiting for position and rotation data...")

        # Add other non-blocking tasks you want to perform here
        # For example, you can add other logic here that runs while waiting for the next transform.

    # Shutdown the node after use
    position_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
