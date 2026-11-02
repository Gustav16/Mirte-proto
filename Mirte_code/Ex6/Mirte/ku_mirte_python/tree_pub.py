import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
import time
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import Pose, Point
import numpy as np

class TreePublisher(Node):
    def __init__(self, topic_name, reference_frame, target_frame='odom', rate=1.0):
        super().__init__('tree_visualizer')
        self.marker_pub = self.create_publisher(Marker, topic_name, 1)
        self.reference_frame = reference_frame
        self.target_frame = target_frame

        self.tree_edges: list = []
        self.edge_colours: list | None = None
        self.edge_widths: list | None = None

        self.timer = self.create_timer(rate, self.publish_markers)  # one-shot
        
        self.tf_buffer = tf2_ros.Buffer() # Create a buffer for the transform
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) # Create a listener for the transform
    
    def clear_previous_markers(self):
        for i in range(len(self.tree_edges) + 10):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = self.target_frame
            marker.ns = "rrt"
            marker.id = i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.DELETE
            
            self.marker_pub.publish(marker)

    def publish_markers(self):
        #print("Publishing markers")
        #print(self.tree_edges)
        if len(self.tree_edges) == 0:
            #self.get_logger().warn("No edges set! Call node.set_markers(edges, colours, widths) to set data.")
            return
        # No longer clear markers here
        for i, ((start, end), colour, width) in enumerate(zip(self.tree_edges, self.edge_colours, self.edge_widths)):
            marker = Marker()
           
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = self.target_frame
           
            marker.ns = "rrt"
            marker.id = i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            
            marker.scale.x = float(width)
            marker.color = ColorRGBA(r=float(colour[0]), g=float(colour[1]), b=float(colour[2]), a=float(colour[3]))

            marker.points.append(start)
            marker.points.append(end)

            self.marker_pub.publish(marker)
    
    def set_markers(self, edges, colours=None, widths=None):
        """
        Set the edges, colors, and widths for the tree visualization.
        edges: List of tuples (start, end) where start and end are (x, y) coordinates.
        colors: List of tuples (r, g, b, a) for each edge.
        widths: List of floats for the width of each edge.
        """
        self.clear_previous_markers()

        edges = [(Point(x=float(start[0]), y=float(start[1]), z=0.0),
                 Point(x=float(end[0]), y=float(end[1]), z=0.0)) for start, end in edges]
        self.tree_edges = [(self.transform_point(start), self.transform_point(end)) for start, end in edges]

        if colours is None:
            self.edge_colours = np.full((len(self.tree_edges), 4), 1.0, dtype=np.float32)
        else:
            self.edge_colours = np.array(colours, dtype=np.float32) / 255.0
        if widths is None:
            self.edge_widths = np.full((len(self.tree_edges), 1), 0.01, dtype=np.float32)
        else:
            self.edge_widths = np.array(widths, dtype=np.float32)

        
    def transform_point(self, position):
        self.get_logger().warn(f"Waiting for transform from {self.reference_frame} to {self.target_frame}")
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
    node = TreePublisher('rrt_markers', 'odom')
    
    try:
        while rclpy.ok():
            
            # Example usage
            edges = [((0, 0), (1, 1)), ((1, 1), (2, 2))]
            colours = [(255, 0, 0, 255), (0, 255, 0, 255)]
            widths = [0.01, 0.02]
            node.set_markers(edges)
            
            rclpy.spin_once(node)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
