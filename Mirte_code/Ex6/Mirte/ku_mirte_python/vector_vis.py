import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, Point, Quaternion

class OdometryPublisher(Node):
    def __init__(self, topic_name, reference_frame, target_frame = 'odom', rate=0.1):
        super().__init__('odometry_publisher')

        self.publisher = self.create_publisher(Odometry, topic_name, 10)
        self.reference_frame = reference_frame
        self.target_frame = target_frame

        self.current_position = None
        self.current_orientation = None

        self.timer = self.create_timer(rate, self.publish_odometry)

        self.tf_buffer = tf2_ros.Buffer() # Create a buffer for the transform
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) # Create a listener for the transform

    def publish_odometry(self):
        odometry_msg = Odometry()

        odometry_msg.header.stamp = self.get_clock().now().to_msg()
        odometry_msg.header.frame_id = self.target_frame

        # Ensure current_position and current_orientation are valid
        odometry_msg.pose.pose.position = self.current_position if self.current_position else Point()
        odometry_msg.pose.pose.orientation = self.current_orientation if self.current_orientation else Quaternion()

        self.publisher.publish(odometry_msg)
    
    def set_position(self, position, orientation):
        if isinstance(position, np.ndarray) or isinstance(position, list):
            position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
        if isinstance(orientation, np.ndarray) or isinstance(orientation, list):
            orientation = Quaternion(x=float(orientation[0]), y=float(orientation[1]), z=float(orientation[2]), w=float(orientation[3]))
        

        self.current_position, self.current_orientation= self.transform_point(position, orientation)
    

    def transform_point(self, position, orientation):
        if self.reference_frame == self.target_frame:
            return position, orientation
        
        try:
            # Wait for the transform to become available
            while not self.tf_buffer.can_transform(self.target_frame, self.reference_frame, rclpy.time.Time().to_msg()):
                rclpy.spin_once(self, timeout_sec=0.1)

            # Get the latest transform from the reference frame to the target frame
            transform = self.tf_buffer.lookup_transform(self.target_frame, self.reference_frame, rclpy.time.Time().to_msg())
            
            # Create a PoseStamped message from the position and orientation
            pose = Pose()
            pose.position = position  # Set position on pose
            pose.orientation = orientation  # Set orientation on pose
            
            # Transform the pose
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)

            # Now, access the position and orientation from the transformed Pose
            return transformed_pose.position, transformed_pose.orientation
        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Transform error: {e}")
            return None, None

    
def main():
    rclpy.init()
    odometry_publisher = OdometryPublisher('test_odometry', 'base_link')

    try:
        while rclpy.ok():
            odometry_publisher.set_position(np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
            # Spin the node to publish the Odometry message
            rclpy.spin_once(odometry_publisher)
    except KeyboardInterrupt:
        ...

    odometry_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()