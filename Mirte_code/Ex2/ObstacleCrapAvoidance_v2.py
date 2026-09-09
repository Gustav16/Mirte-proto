import sys
import os
import math
import time

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../Mirte/ku_mirte_python'
    )
)

from ku_mirte import KU_Mirte


# --------------------------------------------------
# Robot setup
# --------------------------------------------------

mirte = KU_Mirte()

BASE_SPEED_MOD = 2.38
BASE_TURN_MOD = 2.38

mirte.set_driving_modifier(
    BASE_SPEED_MOD,
    BASE_TURN_MOD
)


# --------------------------------------------------
# Movement settings
# --------------------------------------------------

FWD_SPEED = 0.35
APPROACH_SPEED = 0.20
SIDE_SPEED = 0.25

FWD_ANG_SPEED = 0.0

SPIN_RATE = 1.5
TURN_DURATION = (math.pi / 2) / SPIN_RATE

POLL_INTERVAL = 0.05
SCAN_SETTLE = 0.15


# --------------------------------------------------
# Distance settings
# --------------------------------------------------

# Slow down here, but keep moving forward
SLOW_DISTANCE = 0.40

# Only stop when we are quite close to the obstacle
STOP_DISTANCE = 0.20

# When scanning left/right, a side must have at least
# this much free space before we choose it.
SIDE_CLEAR_DISTANCE = 0.35

# While strafing, the obstacle is considered gone from
# the front when BOTH front sonars are farther away than this.
FRONT_CLEAR_DISTANCE = 0.42

CLEAR_READINGS_REQUIRED = 4

# Maximum sideways time, so the robot cannot strafe forever.
MAX_SIDE_TIME = 5.0

# After the obstacle disappears from the front sonar,
# move a little farther sideways to clear its corner.
EXTRA_SIDE_TIME = 0.20

# After moving around the obstacle, drive about 50 cm forward.
PASS_FORWARD_DISTANCE = 0.50

# If the path back to the original line is not clear yet,
# move another 20 cm forward and check again.
RECHECK_FORWARD_DISTANCE = 0.20

# Maximum number of return-path checks.
MAX_RETURN_CHECKS = 5

# Required free space toward the original line.
RETURN_CLEAR_DISTANCE = 0.35

# Same sideways time back should approximately return us
# to the original driving line.
RETURN_SCALE = 1.0


# --------------------------------------------------
# Sonar
# --------------------------------------------------

def read_sonars():
    """
    Read all four sonar sensors.
    """

    sonar = mirte.sonar

    front_left = sonar["front_left"]
    front_right = sonar["front_right"]
    rear_left = sonar["rear_left"]
    rear_right = sonar["rear_right"]

    return {
        "front_left": front_left,
        "front_right": front_right,
        "rear_left": rear_left,
        "rear_right": rear_right,
        "front_clearance": min(front_left, front_right),
        "rear_clearance": min(rear_left, rear_right),
    }


def print_sonars(prefix, s):
    print(
        f"{prefix} | "
        f"FL={s['front_left']:.2f} "
        f"FR={s['front_right']:.2f} | "
        f"RL={s['rear_left']:.2f} "
        f"RR={s['rear_right']:.2f}"
    )


# --------------------------------------------------
# Rotation
# --------------------------------------------------

def turn_90(direction):
    """
    direction:
        +1 = turn 90 degrees left
        -1 = turn 90 degrees right
    """

    mirte.stop()

    mirte.drive(
        0.0,
        direction * SPIN_RATE,
        TURN_DURATION,
        blocking=True
    )

    mirte.stop()
    time.sleep(SCAN_SETTLE)


# --------------------------------------------------
# Scan both sides
# --------------------------------------------------

def scan_left_and_right():
    """
    Robot starts facing its normal forward direction.

    1. Turn 90 degrees left.
    2. Front sonar now looks toward the ORIGINAL left side.
    3. Rear sonar now looks toward the ORIGINAL right side.
    4. Read both directions.
    5. Turn 90 degrees right to face forward again.

    Returns:
        left_clearance,
        right_clearance
    """

    print("\nScanning left and right...")

    turn_90(+1)

    s = read_sonars()
    print_sonars("Side scan", s)

    # After turning left:
    # front points toward original LEFT
    # rear points toward original RIGHT
    left_clearance = s["front_clearance"]
    right_clearance = s["rear_clearance"]

    print(
        f"Left space={left_clearance:.2f} m | "
        f"Right space={right_clearance:.2f} m"
    )

    # Return to original heading
    turn_90(-1)

    return left_clearance, right_clearance


def choose_side():
    """
    Physically scan left/right before strafing.

    Returns:
        +1 = strafe left
        -1 = strafe right
        None = neither side is clear enough
    """

    left_clearance, right_clearance = scan_left_and_right()

    left_ok = left_clearance >= SIDE_CLEAR_DISTANCE
    right_ok = right_clearance >= SIDE_CLEAR_DISTANCE

    if left_ok and right_ok:
        if left_clearance >= right_clearance:
            print("Both sides are clear; choosing LEFT.")
            return +1
        else:
            print("Both sides are clear; choosing RIGHT.")
            return -1

    if left_ok:
        print("Only LEFT is clear enough.")
        return +1

    if right_ok:
        print("Only RIGHT is clear enough.")
        return -1

    print(
        "Neither side has enough free space "
        "for the sideways manoeuvre."
    )

    return None


# --------------------------------------------------
# Normal forward driving
# --------------------------------------------------

def drive_until_close_to_obstacle():
    """
    Drive forward continuously.

    - Full speed when far away.
    - Slow down near the obstacle.
    - Only stop at STOP_DISTANCE.
    """

    current_speed = None

    while True:

        s = read_sonars()
        distance = s["front_clearance"]

        print_sonars("Forward", s)

        if distance <= STOP_DISTANCE:

            mirte.stop()

            print(
                f"\nObstacle is close: {distance:.2f} m."
            )

            return s

        if distance <= SLOW_DISTANCE:
            wanted_speed = APPROACH_SPEED
        else:
            wanted_speed = FWD_SPEED

        if wanted_speed != current_speed:

            mirte.drive(
                wanted_speed,
                FWD_ANG_SPEED,
                None,
                blocking=False
            )

            current_speed = wanted_speed

        time.sleep(POLL_INTERVAL)


# --------------------------------------------------
# Strafe sideways until obstacle disappears
# --------------------------------------------------

def strafe_until_front_clear(direction):
    """
    Move sideways while keeping the robot facing forward.

    IMPORTANT:
    We deliberately do NOT abort because the front sonar is close.
    The obstacle is expected to remain close in front while the robot
    initially moves sideways.

    We continue until the obstacle disappears from the front sonar.

    Returns:
        sideways driving time, or None on timeout.
    """

    side = "left" if direction > 0 else "right"

    print(
        f"\nStrafing {side} until the obstacle "
        "is no longer in front."
    )

    start = time.monotonic()
    clear_readings = 0

    mirte.drive(
        [0.0, direction * SIDE_SPEED],
        0.0,
        None,
        blocking=False
    )

    while True:

        s = read_sonars()
        elapsed = time.monotonic() - start

        print_sonars(f"Strafe {side}", s)

        if s["front_clearance"] >= FRONT_CLEAR_DISTANCE:
            clear_readings += 1
        else:
            clear_readings = 0

        if clear_readings >= CLEAR_READINGS_REQUIRED:

            print(
                "Obstacle has disappeared from the front sonar."
            )

            # Move a little farther sideways to clear the corner.
            time.sleep(EXTRA_SIDE_TIME)

            mirte.stop()

            side_time = time.monotonic() - start

            print(
                f"Total sideways time: {side_time:.2f} s."
            )

            return side_time

        if elapsed >= MAX_SIDE_TIME:

            mirte.stop()

            print(
                "Sideways timeout: obstacle did not "
                "disappear from the front sonar."
            )

            return None

        time.sleep(POLL_INTERVAL)


# --------------------------------------------------
# Drive a requested approximate forward distance
# --------------------------------------------------

def drive_forward_distance(distance):
    """
    Drive approximately 'distance' meters forward.

    Uses:
        time = distance / FWD_SPEED

    Front sonar is still watched for a NEW obstacle.
    """

    duration = distance / FWD_SPEED

    print(
        f"\nDriving forward approximately "
        f"{distance:.2f} m "
        f"({duration:.2f} s)."
    )

    start = time.monotonic()

    mirte.drive(
        FWD_SPEED,
        FWD_ANG_SPEED,
        None,
        blocking=False
    )

    while time.monotonic() - start < duration:

        s = read_sonars()
        print_sonars("Forward bypass", s)

        # This would now be a new obstacle directly ahead.
        if s["front_clearance"] <= STOP_DISTANCE:

            mirte.stop()

            print(
                "New obstacle detected while driving "
                "past the original obstacle."
            )

            return False

        time.sleep(POLL_INTERVAL)

    mirte.stop()

    return True


# --------------------------------------------------
# Check whether path back to original line is clear
# --------------------------------------------------

def return_path_is_clear(original_strafe_direction):
    """
    Physically turn 90 degrees and use front + rear sonar
    to inspect the two sides again.

    If we originally strafed LEFT, the original line is now
    to our RIGHT.

    If we originally strafed RIGHT, the original line is now
    to our LEFT.
    """

    left_clearance, right_clearance = scan_left_and_right()

    if original_strafe_direction > 0:
        # We moved left, so original line is to the right.
        toward_line = right_clearance
        side_name = "right"
    else:
        # We moved right, so original line is to the left.
        toward_line = left_clearance
        side_name = "left"

    print(
        f"Space toward original line ({side_name}) = "
        f"{toward_line:.2f} m"
    )

    return toward_line >= RETURN_CLEAR_DISTANCE


# --------------------------------------------------
# Return to original line
# --------------------------------------------------

def return_to_original_line(direction, side_time):
    """
    Strafe in the opposite direction for approximately
    the same amount of time as the initial sidestep.
    """

    return_direction = -direction
    return_time = side_time * RETURN_SCALE

    side = "left" if return_direction > 0 else "right"

    print(
        f"\nReturning {side} toward original line "
        f"for {return_time:.2f} s."
    )

    mirte.drive(
        [0.0, return_direction * SIDE_SPEED],
        0.0,
        return_time,
        blocking=True
    )

    mirte.stop()

    print(
        "Back approximately on the original driving line."
    )


# --------------------------------------------------
# Complete avoidance manoeuvre
# --------------------------------------------------

def avoid_obstacle():
    """
    Desired sequence:

    1. Stop close to obstacle.
    2. Turn 90 degrees.
    3. Use front + rear sonar to inspect both sides.
    4. Turn 90 degrees back to original heading.
    5. Strafe sideways in the selected direction.
    6. Keep strafing until the front sonar no longer sees obstacle.
    7. Drive approximately 50 cm forward.
    8. Turn 90 degrees and check the sides again.
    9. If the route back is clear, return to original line.
    10. If not clear, drive another 20 cm and check again.
    """

    direction = choose_side()

    if direction is None:

        mirte.stop()
        return False

    # Move sideways without turning the robot.
    side_time = strafe_until_front_clear(direction)

    if side_time is None:

        mirte.stop()
        return False

    # Move about 50 cm forward past the obstacle.
    if not drive_forward_distance(PASS_FORWARD_DISTANCE):

        mirte.stop()
        return False

    # Check whether we can safely move back to the original line.
    for check_number in range(1, MAX_RETURN_CHECKS + 1):

        print(
            f"\nReturn-path check "
            f"{check_number}/{MAX_RETURN_CHECKS}"
        )

        if return_path_is_clear(direction):

            print(
                "Path back to original line is clear."
            )

            return_to_original_line(
                direction,
                side_time
            )

            print(
                "\nObstacle avoidance complete.\n"
            )

            return True

        print(
            "Path back is still blocked."
        )

        # If this was the last allowed check, stop.
        if check_number == MAX_RETURN_CHECKS:

            mirte.stop()

            print(
                "Maximum return checks reached."
            )

            return False

        # Move a little farther forward and try again.
        if not drive_forward_distance(
            RECHECK_FORWARD_DISTANCE
        ):

            mirte.stop()
            return False

    return False


# --------------------------------------------------
# Main
# --------------------------------------------------

try:

    print(
        "Driving forward with scan -> strafe -> "
        "50 cm forward -> scan -> return avoidance."
    )

    print(
        f"Slow at {SLOW_DISTANCE:.2f} m, "
        f"stop at {STOP_DISTANCE:.2f} m."
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    while True:

        drive_until_close_to_obstacle()

        success = avoid_obstacle()

        if not success:

            print(
                "\nCould not complete avoidance safely. "
                "Robot remains stopped."
            )

            break


except KeyboardInterrupt:

    print("\nInterrupted.")


finally:

    mirte.stop()
    del mirte
