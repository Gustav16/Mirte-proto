"""
Closed-loop square driving for Mirte, using wheel encoder feedback
instead of speed*time / speed*seconds dead-reckoning.

Why: commanding "motor at X speed for Y seconds" assumes the motor
actually reaches X speed instantly and stays there exactly, on both
wheels, every run. In practice: the two drive motors are never
perfectly matched, battery voltage sags over a run, there's a short
ramp-up before the motor reaches commanded speed, and wheels slip
(especially during in-place turns). None of that is measured, so none
of it gets corrected -- you end up hand-tuning constants that only
work for one battery level and one floor.

Fix: drive/turn until the ENCODERS say you've gone far enough, not
until a timer says so. This also lets you correct crooked driving
in real time by comparing the two wheels' tick counts.

--- Fill these in for your robot before using ---
"""

import time

# --- Replace with your robot's actual motor/sensor names
# (check your mirte_user_config.yaml or `rostopic list` / the
# mirte_robot API docs for the exact keys your config uses) ---
LEFT_MOTOR = "left"
RIGHT_MOTOR = "right"
LEFT_ENCODER = "left"
RIGHT_ENCODER = "right"

# --- Calibration constants: measure these on your actual robot ---
# 1) Ticks per wheel revolution: spin one wheel exactly one full turn
#    by hand (or drive at low speed and count) and read getEncoder()
#    before/after.
TICKS_PER_REV = 960  # placeholder -- measure this

# 2) Wheel diameter in meters (measure with calipers/ruler).
WHEEL_DIAMETER_M = 0.065  # placeholder

# 3) Wheelbase: distance between the two wheels' contact points with
#    the floor, in meters (measure directly on the chassis).
WHEEL_BASE_M = 0.14  # placeholder

METERS_PER_TICK = (3.14159265 * WHEEL_DIAMETER_M) / TICKS_PER_REV


def drive_distance(robot, distance_m, base_speed=40, k_correct=0.6, poll_hz=50):
    """Drive straight for `distance_m` meters, correcting for the two
    motors not being perfectly matched by comparing encoder deltas."""
    start_left = robot.getEncoder(LEFT_ENCODER)
    start_right = robot.getEncoder(RIGHT_ENCODER)
    target_ticks = distance_m / METERS_PER_TICK

    robot.setMotorControl(True)
    while True:
        d_left = robot.getEncoder(LEFT_ENCODER) - start_left
        d_right = robot.getEncoder(RIGHT_ENCODER) - start_right
        progress = (d_left + d_right) / 2.0

        if progress >= target_ticks:
            break

        # Slow down near the target instead of stopping hard on a
        # timer -- reduces overshoot from momentum.
        remaining = target_ticks - progress
        speed = base_speed if remaining > target_ticks * 0.2 else base_speed * 0.5

        # Proportional correction: if left has gone further than
        # right (or vice versa), nudge the faster one down slightly.
        error = d_left - d_right
        robot.setMotorSpeed(LEFT_MOTOR, speed - k_correct * error)
        robot.setMotorSpeed(RIGHT_MOTOR, speed + k_correct * error)

        time.sleep(1.0 / poll_hz)

    robot.stop()


def turn_angle(robot, angle_rad, speed=35, poll_hz=50):
    """Turn in place by `angle_rad` radians (positive = one direction,
    negative = the other) using encoder ticks, not speed*time."""
    start_left = robot.getEncoder(LEFT_ENCODER)
    start_right = robot.getEncoder(RIGHT_ENCODER)

    # Arc length each wheel must travel for an in-place turn.
    arc_m = abs(angle_rad) * (WHEEL_BASE_M / 2.0)
    target_ticks = arc_m / METERS_PER_TICK

    direction = 1 if angle_rad > 0 else -1
    robot.setMotorControl(True)
    while True:
        d_left = robot.getEncoder(LEFT_ENCODER) - start_left
        d_right = robot.getEncoder(RIGHT_ENCODER) - start_right
        progress = (abs(d_left) + abs(d_right)) / 2.0

        if progress >= target_ticks:
            break

        remaining = target_ticks - progress
        s = speed if remaining > target_ticks * 0.3 else speed * 0.5
        robot.setMotorSpeed(LEFT_MOTOR, -direction * s)
        robot.setMotorSpeed(RIGHT_MOTOR, direction * s)

        time.sleep(1.0 / poll_hz)

    robot.stop()


def drive_square(robot, side_m=0.5):
    for _ in range(4):
        drive_distance(robot, side_m)
        time.sleep(0.2)  # brief settle before turning
        turn_angle(robot, 1.5707963)  # 90 degrees in radians
        time.sleep(0.2)


if __name__ == "__main__":
    # from mirte_robot import robot as mirte
    # r = mirte.createRobot()
    # drive_square(r, side_m=0.5)
    pass