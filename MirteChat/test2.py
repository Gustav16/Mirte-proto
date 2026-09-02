"""
Closed-loop square driving for Mirte, using odometry feedback instead
of speed*time / speed*seconds dead-reckoning.

Why: commanding "drive at X speed for Y seconds" assumes the robot
actually reaches X speed instantly and stays there exactly. In
practice: battery voltage sags over a run, there's a short ramp-up
before commanded speed is reached, and wheels slip (especially during
in-place turns). None of that is measured, so none of it gets
corrected -- you end up hand-tuning constants that only work for one
battery level and one floor.

Fix: drive/turn until ODOMETRY (robot.position / robot.rotation, from
the /odom topic) says you've gone far enough, not until a timer says
so.

Note: the KU_Mirte API is a high-level twist interface (linear m/s +
angular rad/s), not per-wheel motor/encoder access, so there's no
wheel-by-wheel straightness correction here -- odometry already fuses
both wheels for us.
"""

import time
#getting path to import KU_Mirte
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '../Mirte/ku_mirte_python'))

from ku_mirte import KU_Mirte

#init mirte mirte
mirte = KU_Mirte()


def _yaw_from_quaternion(q):
    """Extract the yaw (rotation around Z) in radians from a
    geometry_msgs Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _angle_wrap(angle):
    """Wrap an angle (radians) to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _wait_for_position(robot, timeout_s=5.0, poll_hz=20):
    """Block until the first /odom position update arrives.
    robot.position is None until then, so callers must not read it
    directly before this."""
    deadline = time.time() + timeout_s
    pos = robot.position
    while pos is None:
        if time.time() > deadline:
            raise RuntimeError(
                "No odometry position received from /odom within "
                f"{timeout_s}s -- is the robot publishing odometry?"
            )
        time.sleep(1.0 / poll_hz)
        pos = robot.position
    return pos


def _wait_for_rotation(robot, timeout_s=5.0, poll_hz=20):
    """Block until the first /odom rotation update arrives.
    robot.rotation is None until then, so callers must not read it
    directly before this."""
    deadline = time.time() + timeout_s
    rot = robot.rotation
    while rot is None:
        if time.time() > deadline:
            raise RuntimeError(
                "No odometry rotation received from /odom within "
                f"{timeout_s}s -- is the robot publishing odometry?"
            )
        time.sleep(1.0 / poll_hz)
        rot = robot.rotation
    return rot


def drive_distance(robot, distance_m, speed=0.1, poll_hz=20, timeout_s=30.0):
    """Drive straight for `distance_m` meters, stopping based on the
    robot's actual measured position instead of a timer."""
    start_pos = _wait_for_position(robot)
    slowdown_started = False
    deadline = time.time() + timeout_s

    robot.drive(speed, 0.0, None, blocking=False)
    while True:
        if time.time() > deadline:
            robot.stop()
            raise RuntimeError(
                f"drive_distance: did not reach {distance_m}m within "
                f"{timeout_s}s -- odometry may have stalled or the "
                "robot may be stuck."
            )

        pos = robot.position
        traveled = math.hypot(pos.x - start_pos.x, pos.y - start_pos.y)
        remaining = distance_m - traveled

        if remaining <= 0:
            break

        # Slow down near the target instead of stopping hard on a
        # timer -- reduces overshoot from momentum.
        if not slowdown_started and remaining < distance_m * 0.2:
            robot.drive(speed * 0.5, 0.0, None, blocking=False)
            slowdown_started = True

        time.sleep(1.0 / poll_hz)

    robot.stop()


def turn_angle(robot, angle_rad, speed=0.5, poll_hz=20, timeout_s=30.0):
    """Turn in place by `angle_rad` radians (positive = left, negative
    = right) using measured heading, not speed*time."""
    start_yaw = _yaw_from_quaternion(_wait_for_rotation(robot))
    target = abs(angle_rad)
    direction = 1 if angle_rad > 0 else -1
    slowdown_started = False
    deadline = time.time() + timeout_s

    robot.drive(0.0, direction * speed, None, blocking=False)
    while True:
        if time.time() > deadline:
            robot.stop()
            raise RuntimeError(
                f"turn_angle: did not reach {angle_rad}rad within "
                f"{timeout_s}s -- odometry may have stalled or the "
                "robot may be stuck."
            )

        current_yaw = _yaw_from_quaternion(robot.rotation)
        turned = abs(_angle_wrap(current_yaw - start_yaw))
        remaining = target - turned

        if remaining <= 0:
            break

        if not slowdown_started and remaining < target * 0.3:
            robot.drive(0.0, direction * speed * 0.5, None, blocking=False)
            slowdown_started = True

        time.sleep(1.0 / poll_hz)

    robot.stop()


def drive_square(robot, side_m=0.5):
    for _ in range(4):
        drive_distance(robot, side_m)
        time.sleep(0.2)  # brief settle before turning
        turn_angle(robot, math.pi / 2)  # 90 degrees in radians
        time.sleep(0.2)




try:

    drive_square(mirte, side_m=0.5)

    # ... jeres kode med mirte ...

except KeyboardInterrupt:
    print("Program interrupted!")

finally:
    mirte.stop()
    del mirte
