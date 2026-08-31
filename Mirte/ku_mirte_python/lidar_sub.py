import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import time

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            2  
        )
        self.subscription  # prevent unused variable warning

        self.lidar_message = None
        self.angle_increment: float | None = None
        self.num_points: int | None = None
    
    def lidar_callback(self, msg):
        self.lidar_message = msg

    def ranges(self):
        while self.lidar_message is None:
            self.get_logger().info("Waiting for lidar message...")
            time.sleep(0.1)
        return np.array(self.lidar_message.ranges)

    def angle_section(self, start_angle: float, end_angle: float | None = None):
        """
        Returns the lidar ranges for a given angle section.
        The start_angle and end_angle are in radians from -pi to pi. 
        Here 0 is straight ahead, pi/2 is to the left, and -pi/2 is to the right.
        If only start_angle is given, the function will return the distance at that angle.
        If both start_angle and end_angle is given, the function will return the distances in that section.

        Parameters:
            start_angle (float): The start angle of the section in radians.
            end_angle (float): The end angle of the section in radians.
        """

        if self.lidar_message is None:
            return None
        if self.angle_increment is None:
            self.angle_increment = self.lidar_message.angle_increment
        if self.num_points is None:
            self.num_points = len(self.lidar_message.ranges)
        
        start_index = round(start_angle / self.angle_increment) + (self.num_points // 2)
        
        if end_angle is not None: 
            end_index = round(end_angle / self.angle_increment) + (self.num_points // 2)
            
            if end_angle < start_angle:
                self.get_logger().warn("End angle is smaller than start angle. Swapping start and end angle.")
                start_index, end_index = end_index, start_index
        else:
            end_index = start_index + 1 

        return self.lidar_message.ranges[start_index:end_index]
        
       

def main(args=None):
    rclpy.init(args=args)
    lidar_subscriber = LidarSubscriber()
    rclpy.spin_once(lidar_subscriber)
   
    print(lidar_subscriber.ranges())
    print(lidar_subscriber.angle_section(-3.14 / 4, 3.14 / 4))
    print(lidar_subscriber.angle_section(3.14 / 4, -3.14 / 4))
    print(lidar_subscriber.angle_section(0))
   
    lidar_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()