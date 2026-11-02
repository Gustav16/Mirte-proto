import time
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor

class MovementPublisher(Node):
    """
    This class is used to publish movement commands to the robot.
    Publishes Twist messages to the `/mirte_base_controller/cmd_vel_unstamped` topic.
    """
    def __init__(self, speed_modifier=2.38, rotation_modifier=2.38):
        super().__init__('movement_publisher')
        self.publisher_mirte = self.create_publisher(Twist, '/mirte_base_controller/cmd_vel', 10)
        self.publisher_gazebo = self.create_publisher(Twist, '/mirte_base_controller/cmd_vel_unstamped', 10)

        self.lin_speed: float|list = 0.0
        self.ang_speed: float = 0.0
        self.duration: float | None = 0.0

        # Multiplier for the speed of the robot between the Gazebo and Mirte controller.
        self.speed_modifier: float = speed_modifier  # Speed modifier to adjust the speed of the robot
        self.rotation_modifier: float = rotation_modifier  # Rotation modifier to adjust the rotation speed of the robot

        self.start_drive_time: float = time.time()
        self.driving : bool = False

        self.timer = self.create_timer(0.05, self._publish_volicity) 

    def drive(self, lin_speed: list|float, ang_speed: float, duration: float, interrupt:bool=True):
        """
        Drive the robot with a given speed and direction for a given duration.
        If duration is None, the robot will drive forever.
        If interrupt is `True`, the current drive will be interrupted. 
        If it however is `False`, the current drive will not be interrupted.

        Parameters:
            lin_speed (float | list): The linear velocity (m/s) of the robot. Positive values drive forward, negative values drive backward. Can either be a float or a list with two floats. Using float will drive the robot in the x direction, using a list will drive the robot in the x and y direction.
            ang_speed (float): The angular velocity (rad/s) of the robot. Positive values turn left, negative values turn right.
            duration (float): The duration (seconds) of the drive. If `None`, the robot will drive forever.
            interrupt (bool): If `True`, the current drive will be interrupted. If `False`, the current drive will not be interrupted.
        """
        if self.driving:
            if not interrupt: # The robot is already driving and dont want to interrupt
                return # Did not change the drive of the robot
        
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.duration = duration
        self.start_drive_time = time.time()        

    def stop(self):
        """
        Stops the robot.
        """
        # Stop the robot by publishing zero velocities
        self.driving = False
        self.duration = 0.0
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = 0.0
        self.publisher_mirte.publish(twist)
        self.publisher_gazebo.publish(twist)

    def _publish_volicity(self):
        """
        Always running. 
        """
        if self.duration is not None:
            if self.duration == 0.0:
                self.stop()
                return
            if time.time() - self.start_drive_time > self.duration:
                self.stop()
                return

        self.driving = True
        twist = Twist()

        if isinstance(self.lin_speed, list):
            if len(self.lin_speed) != 2:
                raise ValueError("lin_speed must be a float or a list of length 2")
            twist.linear.x = float(self.lin_speed[0])
            twist.linear.y = float(self.lin_speed[1])
        else:
            if not isinstance(self.lin_speed, (int, float)):
                raise ValueError("lin_speed must be a float or a list of length 2")
            twist.linear.x = float(self.lin_speed)
        twist.angular.z = float(self.ang_speed)

        self.publisher_gazebo.publish(twist)
        twist.linear.x *= self.speed_modifier
        twist.linear.y *= self.speed_modifier
        twist.angular.z *= self.rotation_modifier
        self.publisher_mirte.publish(twist)


def main():
    print("Initializing")
    rclpy.init()
    node = MovementPublisher()
    
    print("Starting thread")
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Run the executor in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    node.drive(-0.5, 0.0, 1.0)
    
    time.sleep(0.5) 

    node.drive(0.5, 0.0, 3.0)

    time.sleep(1)

    node.drive([0.0,1.0], 0.0, 3.0)

    time.sleep(2)

    print("Stoppig")
    node.stop()
    executor_thread.join()  # Wait for the thread to finish before proceeding
    
    # Ensure proper shutdown
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
