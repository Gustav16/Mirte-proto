import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import struct
import numpy as np
import time
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, Point


class PointCloudPublisher(Node):
    def __init__(self, topic_name, reference_frame, target_frame = 'odom', rate=0.2):
        super().__init__('pointcloud_publisher')
        
        self.publisher_ = self.create_publisher(PointCloud2, topic_name, 1)
        self.reference_frame = reference_frame
        self.target_frame = target_frame
        
        self.current_points = np.empty((0, 3))  # Default empty point cloud
        self.current_colors = np.empty((0, 4))  # Default empty color array

        self.timer = self.create_timer(rate, self.publish_pointcloud)

        self.tf_buffer = tf2_ros.Buffer() # Create a buffer for the transform
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) # Create a listener for the transform

    

    def publish_pointcloud(self):
        if self.current_points.size == 0:
            #self.get_logger().warn("No points set! Call node.set_points(points, colors) to set data.")
            return

        cloud_msg = PointCloud2()

        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = self.target_frame  

        cloud_msg.height = 1
        cloud_msg.width = len(self.current_points)
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgba', offset=12, datatype=PointField.FLOAT32, count=1)  # Color field
        ]
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16  # 4 bytes per float * 4 fields (x, y, z, rgba)
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True

        # Pack the data into the PointCloud2 message
        data = []
        for point, color in zip(self.current_points, self.current_colors):
            r, g, b, a = color  # Extract RGBA
            rgba = struct.unpack('f', struct.pack('BBBB', b, g, r, a))[0]  # Pack color into float32
            data.append(struct.pack('ffff', point.x, point.y, point.z, rgba))  # Pack all fields

        cloud_msg.data = b''.join(data)

        self.publisher_.publish(cloud_msg)
        # self.get_logger().info(f'Published PointCloud2 with {len(self.current_points)} points')

    def set_points(self, points, colours=None):
        """Update the point cloud data dynamically, with optional colors."""

        points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
        
        transformed_points = [self.transform_point(point) for point in points]
        self.current_points = np.array(transformed_points)

        # If colors are not provided, set default white (RGBA = 255,255,255,255)
        if colours is None:
            self.current_colors = np.full((len(self.current_points), 4), 255, dtype=np.uint8)
        else:
            self.current_colors = np.array(colours, dtype=np.uint8)

        # self.get_logger().info(f"Updated point cloud with {len(self.current_points)} points.")

    def transform_point(self, position):
        if self.reference_frame == self.target_frame:
            return position
        
        try:
            # Wait for the transform to become available
            while not self.tf_buffer.can_transform(self.target_frame, self.reference_frame, rclpy.time.Time().to_msg()):
                #rclpy.spin_once(self, timeout_sec=0.1)
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

def main():
    rclpy.init()
    node = PointCloudPublisher('test_pointcloud', 'base_link')

    try:
        while rclpy.ok():
            # Generate new random points
            new_points = np.random.rand(100, 3) * 10

            # Generate random colors (RGBA format, values 0-255)
            new_colors = np.random.randint(0, 256, (100, 4), dtype=np.uint8)

            node.set_points(new_points, new_colors)  # Update the point cloud with colors
            rclpy.spin_once(node, timeout_sec=1.0)  # Process callbacks
            time.sleep(0.05)  # Wait before updating again

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
