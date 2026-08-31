"""
KU_Mirte class and related methods for robot control and sensor access in ROS2.

This module provides the KU_Mirte class, which acts as a high-level interface for controlling a Mirte robot in a ROS2 environment. It includes methods for driving, sensor access, and publishing various data types such as odometry, trees, occupancy grids, and point clouds.

Classes:
    KU_Mirte: Main class for robot control and sensor access.
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
import numpy as np
import time
from robot_pos_sub import PositionSubscriber
from camera_sub import CameraSubscriber
from camera_compressed_sub import CameraCompressedSubscriber
from lidar_sub import LidarSubscriber
from sonar_sub import SonarSubscriber
from vector_vis import OdometryPublisher
from point_cloud_vis import PointCloudPublisher
from drive_pub import MovementPublisher
from tree_pub import TreePublisher
from map_pub import OccupancyMapPublisher

class KU_Mirte:
    """
    High-level interface for controlling a Mirte robot and accessing its sensors in ROS2.

    This class initializes all required ROS2 nodes and provides methods for driving the robot, accessing sensor data, and publishing visualization data.
    """
    def __init__(self):
        """
        Initializes all robot components, ROS2 nodes, and starts the executor thread.
        Sets up publishers and subscribers for position, camera, lidar, sonar, odometry, tree, point cloud, and occupancy grid.
        """
        print("Initiating components")
        rclpy.init()
        self.robot_pos_sub = PositionSubscriber()
        # self.camera_sub = CameraSubscriber()
        self.camera_compressed_sub = CameraCompressedSubscriber()
        self.lidar_sub = LidarSubscriber()
        self.sonar_sub = SonarSubscriber()
        self.odometry_pub_mirte = OdometryPublisher('odometry_mirte', 'base_link')
        self.odometry_pub_world = OdometryPublisher('odometry_world', 'odom')
        # self.tree_pub_mirte = TreePublisher('tree_mirte', 'base_link')
        # self.tree_pub_world = TreePublisher('tree_world', 'odom')
        # self.pointcloud_pub_mirte = PointCloudPublisher('pointcloud_mirte', 'base_link')
        # self.pointcloud_pub_world = PointCloudPublisher('pointcloud_world', 'odom')
        # self.occupancy_pub_mirte = OccupancyMapPublisher('occupancy_grid_mirte', 'base_link')
        self.movement_pub = MovementPublisher()

        self.executor_thread = None
        self.executor = None
        self._start_executor_thread()

        rclpy.spin_once(self.robot_pos_sub, timeout_sec=2.0)
        #rclpy.spin_once(self.lidar_sub)

        # self.k_matrix = self.camera_sub.k_matrix # Camera intrinsic matrix
        # self.d_matrix = self.camera_sub.d_matrix # Camera distortion
        # self.p_matrix = self.camera_sub.p_matrix # Camera projection matrix

        # # Check if k_matrix is set, if not, set it to a default value
        # if self.k_matrix is None or np.all(self.k_matrix == 0) or self.k_matrix.shape != (3, 3):
        #     self.k_matrix = np.array([[590.0, 0.0, 320.0],
        #                               [0.0, 590.0, 240.0],
        #                               [0.0, 0.0, 1.0]])
        # # Check if d_matrix is set, if not, set it to a default value
        # if self.d_matrix is None or np.all(self.d_matrix == 0) or self.d_matrix.shape != (5,):
        #     self.d_matrix = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    

        print("KU Mirte initialized.")

    def __del__(self):
        """
        Ensures that all ROS2 nodes are properly destroyed and the executor is shut down when the object is deleted.
        """
        self._stop_executor_thread()

        if hasattr(self, 'robot_pos_sub') and self.robot_pos_sub:
            self.robot_pos_sub.destroy_node()
        if hasattr(self, 'camera_sub') and self.camera_sub:
            self.camera_sub.destroy_node()
        if hasattr(self, 'camera_compressed_sub') and self.camera_compressed_sub:
            self.camera_compressed_sub.destroy_node()
        if hasattr(self, 'lidar_sub') and self.lidar_sub:
            self.lidar_sub.destroy_node()
        if hasattr(self, 'sonar_sub') and self.sonar_sub:
            self.sonar_sub.destroy_node()
        if hasattr(self, 'odometry_pub_mirte') and self.odometry_pub_mirte:
            self.odometry_pub_mirte.destroy_node()
        if hasattr(self, 'odometry_pub_world') and self.odometry_pub_world:
            self.odometry_pub_world.destroy_node()
        if hasattr(self, 'pointcloud_pub_mirte') and self.pointcloud_pub_mirte:
            self.pointcloud_pub_mirte.destroy_node()
        if hasattr(self, 'pointcloud_pub_world') and self.pointcloud_pub_world:
            self.pointcloud_pub_world.destroy_node()
        if hasattr(self, 'movement_pub') and self.movement_pub:
            self.movement_pub.destroy_node()
        if hasattr(self, 'occupancy_pub_mirte') and self.occupancy_pub_mirte:
            self.occupancy_pub_mirte.destroy_node()
        if hasattr(self, 'tree_pub_mirte') and self.tree_pub_mirte:
            self.tree_pub_mirte.destroy_node()
        
        rclpy.shutdown()
    
    
    def _start_executor_thread(self):
        """
        Starts the ROS2 MultiThreadedExecutor in a separate thread and adds all relevant nodes.
        """
        print("Starting executor thread...")
        self.executor = MultiThreadedExecutor()

        # self.executor.add_node(self.camera_sub)
        self.executor.add_node(self.camera_compressed_sub)
        self.executor.add_node(self.movement_pub)
        self.executor.add_node(self.lidar_sub)
        self.executor.add_node(self.sonar_sub)
        # self.executor.add_node(self.pointcloud_pub_mirte)
        # self.executor.add_node(self.pointcloud_pub_world)
        # self.executor.add_node(self.occupancy_pub_mirte)
        # self.executor.add_node(self.tree_pub_mirte)

        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        print("Executor thread started.")
    
    def _stop_executor_thread(self):
        """
        Stops the executor thread and shuts down the executor if running.
        """
        if hasattr(self, 'executor_thread') and self.executor_thread:
            self.executor.shutdown()
            self.executor_thread.join()
            self.executor = None
            self.executor_thread = None

    def set_driving_modifier(self, speed_modifier:float = 2.38, turn_modifier:float = 2.38):
        """
        Sets the speed and rotation modifier for the robot's driving behavior. 
        Is multiplied with the linear and angular speed when driving. 
        This is only applied to real driving, not in simulation as the simulation is accurate.
        Args:
            speed_modifier (float): Multiplier for linear speed. 1.0 is normal speed.
            turn_modifier (float): Multiplier for angular speed. 1.0 is normal rotation.
        """
        self.movement_pub.speed_modifier = float(speed_modifier)
        self.movement_pub.rotation_modifier = float(turn_modifier)


    def drive(self, lin_speed:float|list, ang_speed:float, duration:float|None, blocking:bool=True, interrupt:bool=True):
        """
        Drives the robot with a given linear and angular speed for a specified duration.
        Speeds are in meters per second (m/s) for linear speed and radians per second (rad/s) for angular speed.
        The distance driven is calculated as d = lin_speed * duration, 
        and the angle turned is calculated as r = ang_speed * duration (in radians).
        For example, if ang_speed is 0.5 rad/s and duration is 2*pi seconds, the robot will turn pi radians (180 degrees).
        Args:
            lin_speed (float): Linear velocity (m/s). Positive is forward, negative values drive backward.  Can either be a float or a list with two floats. Using float will drive the robot in the x direction, using a list will drive the robot in the x and y direction.
            ang_speed (float): Angular velocity (rad/s). Positive is left.
            duration (float): Duration in seconds. `None` means drive indefinitely.
            blocking (bool): If `True`, waits for drive to finish before returning.
            interrupt (bool): If `True`, interrupts any current drive. If `False`, does not interrupt the current drive.
        """
        self.movement_pub.drive(lin_speed, ang_speed, duration, interrupt=interrupt)
        
        time.sleep(0.2) # Allow time for the thread to realize it is driving
        while blocking and self.movement_pub.driving: # Blocking
            time.sleep(0.01) # Robot can stop during this sleep, it is only to prevent the function from returning before the drive is finished.
    
    def stop(self):
        """
        Stops the robot's movement immediately.
        """
        self.movement_pub.stop()

    @property
    def is_driving(self) -> bool:
        """
        Returns True if the robot is currently driving, False otherwise.
        Returns:
            bool: Driving state.
        """
        return self.movement_pub.driving

    def get_position(self):
        """
        Returns the most recent position of the robot as a tuple or array.
        Returns:
            The robot's position. Use .x, .y, .z for individual coordinates.
        """
        rclpy.spin_once(self.robot_pos_sub, timeout_sec = 2.0)
        return self.robot_pos_sub.get_position()
    
    @property
    def position(self):
        """
        Property that updates and returns the robot's position.
        Returns:
            The robot's position. Use .x, .y, .z for individual coordinates.
        """
        position = self.get_position()
        if position is None:
            print("Position not initialized...")
        return position
 
    def get_rotation(self) -> float:
        """
        Returns the most recent rotation of the robot.
        Returns:
            The robot's rotation in radians.
        """
        rclpy.spin_once(self.robot_pos_sub, timeout_sec = 2.0)
        return self.robot_pos_sub.get_rotation()
    
    @property
    def rotation(self) -> float:
        """
        Property that updates and returns the robot's rotation.
        Returns:
            The robot's rotation in radians
        """
        rotation = self.get_rotation()
        if rotation is None:
            print("Rotation not initialized...")
        return rotation

    def get_image(self) -> np.ndarray:
        """
        Returns the most recent image from the robot's camera.
        Returns:
            Image data (numpy array)
        """
        #rclpy.spin_once(self.camera_sub)
        return self.camera_sub.image()
    
    def get_image_compressed(self) -> np.ndarray:
        """
        Returns the most recent image from the robot's camera (compressed version).
        This is preferred when reading from a remote machine whose connection is low quality, e.g. through WIFI
        Returns:
            Image data (numpy array)
        """
        #rclpy.spin_once(self.camera_sub)
        return self.camera_compressed_sub.image()         
    
    @property
    def image(self) -> np.ndarray:
        """
        Returns the most recent image from the robot's camera.
        Returns:
            Image data (numpy array)
        """
        return self.get_image()
    
    def get_lidar_ranges(self) -> list:
        """
        Returns the most recent lidar data. 
        Is a list of floats representing distances in meters.
        They are evenly spaced in a circle, so the angular distance between each point is 2*pi / len(ranges).
        Returns:
            Lidar data (list of floats).
        """
        #rclpy.spin_once(self.lidar_sub)
        return self.lidar_sub.ranges()

    @property
    def lidar(self) -> list:
        """
        Returns the most recent lidar data.
        Is a list of floats representing distances in meters.
        They are evenly spaced in a circle, so the angular distance between each point is 2*pi / len(ranges).
        Returns:
            Lidar data (list of floats).
        """
        return self.get_lidar_ranges()
        
    def get_lidar_section(self, start_angle:float, end_angle:float) -> list:
        """
        Returns the lidar ranges for a given angle section.
        The angles are in radians, where -pi is the left side, 0 is forward, and pi is right side.
        Args:
            start_angle (float): Start angle in radians (-pi to pi).
            end_angle (float): End angle in radians (-pi to pi).
        Returns:
            Section of lidar range data (list of floats).
        """
        #rclpy.spin_once(self.lidar_sub)
        return self.lidar_sub.angle_section(start_angle, end_angle)

    def get_sonar_ranges(self) -> dict:
        """
        Returns the most recent sonar range data.
        Returns:
            Sonar range data as a dictionary of floats in meters with keys 'front_left', 'front_right', 'rear_left', 'rear_right'.
        """
        #rclpy.spin_once(self.sonar_sub)
        return self.sonar_sub.get_distances()

    @property
    def sonar(self) -> dict:
        """
        Returns the most recent sonar range data.
        Returns:
            Sonar range data as a dictionary of floats in meters with keys 'front_left', 'front_right', 'rear_left', 'rear_right'.
        """
        return self.get_sonar_ranges()

    def set_odometry(self, reference: str, position: list, rotation:list):
        """
        Sets the odometry position and rotation for the robot.
        The reference can be either 'mirte' or 'world', which determines the coordinate frame used.
        Args:
            reference (str): 'mirte' or 'world'. If 'mirte', sets the position in the Mirte frame. If 'world', sets the position in the world frame.
            position: Position data like [x, y, z] 
            rotation: Rotation data as quaternion in list [x, y, z, w].
        Raises:
            ValueError: If reference is not 'mirte' or 'world'.
        """
        if reference == 'mirte':
            self.odometry_pub_mirte.set_position(position, rotation)
            rclpy.spin_once(self.odometry_pub_mirte)
        elif reference == 'world':
            self.odometry_pub_world.set_position(position, rotation)
            rclpy.spin_once(self.odometry_pub_world)
        else:
            raise ValueError("Reference must be 'mirte' or 'world'.")
    
    def set_tree(self, reference:str, edges:list, colours:list|None=None, widths:list|None=None):
        """
        Sets the tree visualization data for the robot.
        Args:
            reference (str): 'mirte' or 'world'. If 'mirte', sets the tree in the Mirte frame. If 'world', sets the tree in the world frame.
            edges: Tree edge data. Should be a list of tuples or list of lists, where each tuple/list contains two points (start and end) as (x, y) coordinates. For example: [([1, 2], [3, 4]), ([3, 4], [5, 6])] will make an edge from (1, 2) to (3, 4) and another from (3, 4) to (5, 6).
            colours: Optional color data. If not provided, defaults to white (255, 255, 255, 255). Otherwise, should be the same length as edges and contain (r, g, b, a) values for each edge in range [0,255].
            widths: Optional width of each edge. If not provided, defaults to 0.01 for all edges. Should be the same length as edges.
        Raises:
            ValueError: If reference is not 'mirte' or 'world'.
        """
        if reference == 'mirte':
            self.tree_pub_mirte.set_markers(edges, colours, widths)
        elif reference == 'world':
            self.tree_pub_world.set_markers(edges, colours, widths)
        else:
            raise ValueError("Reference must be 'mirte' or 'world'.")
        
    def set_occupancy_grid(self, grid:list, resolution:float, origin:tuple=(0.0, 0.0), rotation:float=1.0):
        """
        Visualizes the occupancy grid data. 
        Args:
            grid: Occupancy grid as a 2D list or numpy array. Value is a color grayscale value from 0 to 100, where 0 is free space, 100 is occupied, and -1 is unknown.
            resolution: Grid resolution. This is the size of each cell in meters. For example, a resolution of 0.1 means each cell is 10cm x 10cm.
            origin (tuple): Origin of the grid (default (0.0, 0.0)) which is Mirte's position in the world frame.
            rotation (float): Grid rotation (default 1.0). Rotates around Mirte's orientation.
        """
        print(f"rotation: {rotation}")
        self.occupancy_pub_mirte.set_grid(grid, resolution, origin, rotation=rotation)
    
    def set_pointcloud(self, reference:str, points:list, colors:list=None):
        """
        Sets the point cloud data for the robot.
        Args:
            reference (str): 'mirte' or 'world'. If 'mirte', sets the point cloud in the Mirte frame. If 'world', sets the point cloud in the world frame.
            points: Point cloud data. Should be a list of tuples or lists, where each tuple/list contains three coordinates (x, y, z). 
            colors: Optional color data. If not provided, defaults to white (255, 255, 255, 255). Otherwise, should be the same length as points and contain (r, g, b, a) values for each point in range [0,255].
        Raises:
            ValueError: If reference is not 'mirte' or 'world'.
        """
        if reference == 'mirte':
            self.pointcloud_pub_mirte.set_points(points, colors)
            #rclpy.spin_once(self.pointcloud_pub_mirte)
        elif reference == 'world':
            self.pointcloud_pub_world.set_points(points, colors)
            #rclpy.spin_once(self.pointcloud_pub_world)
        else:
            raise ValueError("Reference must be 'mirte' or 'world'.")

