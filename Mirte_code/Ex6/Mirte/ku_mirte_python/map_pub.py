import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, Point
import time
import numpy as np

class OccupancyMapPublisher(Node):
    def __init__(self, topic_name, reference_frame, target_frame='odom', rate = 1.0):
        super().__init__('occupancy_map_publisher')
        
        self.occupancy_pub = self.create_publisher(OccupancyGrid, topic_name, 10)
        self.target_frame = target_frame
        self.reference_frame = reference_frame

        # Example occupancy map: 10x10 with a cross pattern
        self.grid = []
        self.resolution = 0.1  # 10 cm resolution
        self.origin = (0.0, 0.0)  # Origin in the map frame
        self.rotation = 1.0  # No rotation

        self.timer = self.create_timer(rate, self.generate_occupancy_grid) 

        self.tf_buffer = tf2_ros.Buffer() # Create a buffer for the transform
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) # Create a listener for the transform


    def generate_occupancy_grid(self):
        if len(self.grid) == 0:
            #self.get_logger().warn("No grid set! Call node.set_grid(grid, resolution, origin, rotation) to set data.")
            return

        height, width = self.grid.shape

        transformed_origin = self.transform_point(Point(x=self.origin[0], y=self.origin[1], z=0.0))

        msg = OccupancyGrid()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.resolution
        msg.info.width = width
        msg.info.height = height
        # Origin in middle of the map
        msg.info.origin.position.x = transformed_origin.x - (width * self.resolution) / 2
        msg.info.origin.position.y = transformed_origin.y - (height * self.resolution) / 2
        msg.info.origin.position.z = 0.0
        #print(f"orientation: {self.rotation}")
        msg.info.origin.orientation.w = self.rotation

        msg.data = self.grid.flatten().tolist()  # Flatten the grid to a 1D list
        self.occupancy_pub.publish(msg)

    def set_grid(self, grid, resolution, origin=(0.0, 0.0), rotation=1.0):
        self.grid = np.array(grid, dtype=np.int8)
        self.resolution = float(resolution)
        self.origin = origin
        self.rotation = rotation

    def transform_point(self, position):
        if self.reference_frame == self.target_frame:
            return position
        try:
            # Wait for the transform to become available
            while not self.tf_buffer.can_transform(self.target_frame, self.reference_frame, rclpy.time.Time().to_msg()):
                #rclpy.spin_once(self, timeout_sec=0.1)
                self.get_logger().warn(f"Waiting for transform from {self.reference_frame} to {self.target_frame}")
                time.sleep(0.1)

            # Get the latest transform from the reference frame to the target frame
            transform = self.tf_buffer.lookup_transform(self.target_frame, self.reference_frame, rclpy.time.Time().to_msg())
            
            # Create a PoseStamped message from the position and orientation
            pose = Pose()
            pose.position = position
            
            # Transform the pose to the target frame
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
            
            return transformed_pose.position

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Transform error: {e}")
            return None, None

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyMapPublisher('odom')
    node.set_grid([[0, 100, 0], [100, 100, 100], [0, 100, 0]], 1.0)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
