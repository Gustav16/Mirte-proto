import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
import numpy as np

class SonarSubscriber(Node):
    def __init__(self):
        super().__init__('sonar_subscriber')
        
        """
        Sonars on:
        /io/distance/front_left
        /io/distance/front_right
        /io/distance/rear_left
        /io/distance/rear_right
        """

        self.sub_front_left = self.create_subscription(
            Range,
            '/io/distance/front_left',
            self.front_left_callback,
            1)
        self.sub_front_right = self.create_subscription(
            Range,
            '/io/distance/front_right',
            self.front_right_callback,
            1)
        self.sub_rear_left = self.create_subscription(
            Range,
            '/io/distance/rear_left',
            self.rear_left_callback,
            1)
        self.sub_rear_right = self.create_subscription(
            Range,
            '/io/distance/rear_right',
            self.rear_right_callback,
            1)
        
        self.front_left_distance: float = np.inf
        self.front_right_distance: float = np.inf
        self.rear_left_distance: float = np.inf
        self.rear_right_distance: float = np.inf

    def front_left_callback(self, msg):
        self.front_left_distance = msg.range
    
    def front_right_callback(self, msg):
        self.front_right_distance = msg.range

    def rear_left_callback(self, msg):
        self.rear_left_distance = msg.range

    def rear_right_callback(self, msg):
        self.rear_right_distance = msg.range
    
    def get_distances(self):
        """
        Returns the distances from the sonar sensors.
        """
        return {
            'front_left': self.front_left_distance,
            'front_right': self.front_right_distance,
            'rear_left': self.rear_left_distance,
            'rear_right': self.rear_right_distance
        }
    
    @property
    def front_left(self):
        return self.front_left_distance
    @property
    def front_right(self):
        return self.front_right_distance
    @property
    def rear_left(self):
        return self.rear_left_distance
    @property
    def rear_right(self):
        return self.rear_right_distance
    @property
    def front(self):
        return (self.front_left_distance + self.front_right_distance) / 2
    @property
    def rear(self):
        return (self.rear_left_distance + self.rear_right_distance) / 2
