import sys
import os
import time

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../Mirte/ku_mirte_python'
    )
)

from ku_mirte import KU_Mirte


# -------------------------
# Settings
# -------------------------

# Normal forward movement
FWD_SPEED = 0.35
FWD_ANG_SPEED = 0.0

# Sideways movement with Mecanum wheels
SIDE_SPEED = 0.25

# Robot driving modifiers
BASE_SPEED_MOD = 2.38
BASE_TURN_MOD = 2.38

# Obstacle detection
SAFE_DISTANCE = 0.35

# Require a little more distance before considering
# the obstacle completely cleared.
CLEAR_DISTANCE = 0.45

POLL_INTERVAL = 0.05

# Require several clear measurements in a row.
# This prevents one bad sonar reading from ending the sidestep.
CLEAR_READINGS_REQUIRED = 4

# Maximum allowed sideways movement.
# Safety against getting stuck strafing forever.
MAX_SIDE_TIME = 4.0

# Once the obstacle disappears from the sonar,
# continue sideways a little bit more for safety.
EXTRA_SIDE_TIME = 0.30

# How long to drive forward while passing the obstacle.
BYPASS_TIME = 3.0

# Ideally, returning for the same amount of time
# brings us back onto the original line.
# Tune this after testing on the real robot if needed.
RETURN_SCALE = 1.0


# -------------------------
# Robot setup
# -------------------------

mirte = KU_Mirte()

mirte.set_driving_modifier(
    BASE_SPEED_MOD,
    BASE_TURN_MOD
)


# -------------------------
# Sensor helpers
# -------------------------

def front_clearance():
    """
    Returns:
        minimum distance,
        front left distance,
        front right distance
    """

    sonar = mirte.sonar

    front_left = sonar["front_left"]
    front_right = sonar["front_right"]

    minimum = min(
        front_left,
        front_right
    )

    return minimum, front_left, front_right


# -------------------------
# Forward movement
# -------------------------

def drive_until_blocked():
    """
    Drive straight forward until an obstacle is detected.

    Robot is stopped before returning.

    Returns:
        front_left,
        front_right
    """

    mirte.drive(
        FWD_SPEED,
        FWD_ANG_SPEED,
        None,
        blocking=False
    )

    while True:

        clearance, front_left, front_right = front_clearance()

        print(
            f"Forward | "
            f"L={front_left:.2f} m "
            f"R={front_right:.2f} m"
        )

        if clearance < SAFE_DISTANCE:

            mirte.stop()

            print("Obstacle detected.")

            return front_left, front_right

        time.sleep(POLL_INTERVAL)


# -------------------------
# Sideways movement
# -------------------------

def strafe_until_clear(direction):
    """
    Move sideways until the obstacle disappears
    from the front sonar sensors.

    direction:
        +1 = left
        -1 = right

    Returns:
        total amount of time spent moving sideways,
        or None if the obstacle could not be cleared safely.
    """

    side = "left" if direction > 0 else "right"

    print(f"Moving sideways to the {side}.")

    start_time = time.time()
    clear_readings = 0

    mirte.drive(
        [0.0, direction * SIDE_SPEED],
        0.0,
        None,
        blocking=False
    )

    while True:

        clearance, front_left, front_right = front_clearance()
        elapsed = time.time() - start_time

        print(
            f"Sideways {side} | "
            f"L={front_left:.2f} m "
            f"R={front_right:.2f} m"
        )

        # Obstacle no longer in front
        if clearance > CLEAR_DISTANCE:
            clear_readings += 1
        else:
            clear_readings = 0

        # Require several clear readings
        if clear_readings >= CLEAR_READINGS_REQUIRED:

            print("Obstacle no longer visible in front.")

            # Keep moving sideways slightly more
            # to create a safety margin.
            time.sleep(EXTRA_SIDE_TIME)

            mirte.stop()

            side_time = time.time() - start_time

            print(
                f"Sideways movement took "
                f"{side_time:.2f} seconds."
            )

            return side_time

        # Safety limit
        if elapsed > MAX_SIDE_TIME:

            mirte.stop()

            print(
                "Could not clear obstacle "
                "within maximum sideways time."
            )

            return None

        time.sleep(POLL_INTERVAL)


# -------------------------
# Drive past obstacle
# -------------------------

def drive_past_obstacle():
    """
    Drive straight forward alongside / past the obstacle.

    Still checks the front sonars while driving.

    Returns:
        True if the bypass completed,
        False if another obstacle was detected.
    """

    print("Driving forward past obstacle.")

    mirte.drive(
        FWD_SPEED,
        FWD_ANG_SPEED,
        None,
        blocking=False
    )

    start_time = time.time()

    while time.time() - start_time < BYPASS_TIME:

        clearance, front_left, front_right = front_clearance()

        if clearance < SAFE_DISTANCE:

            mirte.stop()

            print(
                "Obstacle detected while "
                "passing the first obstacle."
            )

            return False

        time.sleep(POLL_INTERVAL)

    mirte.stop()

    print("Obstacle should now be passed.")

    return True


# -------------------------
# Return to original line
# -------------------------

def return_to_line(direction, side_time):
    """
    Move sideways in the opposite direction for
    approximately the same amount of time.

    This should bring the robot back onto its
    original driving line.
    """

    return_direction = -direction
    return_time = side_time * RETURN_SCALE

    side = "left" if return_direction > 0 else "right"

    print(
        f"Returning {side} for "
        f"{return_time:.2f} seconds."
    )

    mirte.drive(
        [0.0, return_direction * SIDE_SPEED],
        0.0,
        return_time,
        blocking=True
    )

    mirte.stop()

    print("Back approximately on original line.")


# -------------------------
# Obstacle avoidance
# -------------------------

def avoid_obstacle(front_left, front_right):
    """
    1. Choose the side with most apparent space.
    2. Move sideways until obstacle disappears.
    3. Drive forward past obstacle.
    4. Move sideways back the same approximate distance/time.

    Returns:
        True if avoidance completed successfully.
        False if the robot could not safely complete the maneuver.
    """

    # +1 = left
    # -1 = right
    if front_left >= front_right:
        direction = 1
        side = "left"
    else:
        direction = -1
        side = "right"

    print(
        f"\nObstacle:"
        f" L={front_left:.2f} m"
        f" R={front_right:.2f} m"
    )

    print(f"More room on the {side}.")

    # Step 1: move sideways until obstacle is no longer in front
    side_time = strafe_until_clear(direction)

    if side_time is None:
        mirte.stop()
        print("Avoidance failed. Robot stays stopped.")
        return False

    # Step 2: drive forward past obstacle
    passed = drive_past_obstacle()

    if not passed:
        mirte.stop()
        print(
            "Could not safely pass obstacle. "
            "Robot stays stopped."
        )
        return False

    # Step 3: return to approximately the original line
    return_to_line(
        direction,
        side_time
    )

    print(
        "Avoidance complete. "
        "Continuing on original heading.\n"
    )

    return True


# -------------------------
# Main
# -------------------------

try:

    print(
        "Driving forward with "
        "Mecanum sideways obstacle avoidance."
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    while True:

        front_left, front_right = drive_until_blocked()

        success = avoid_obstacle(
            front_left,
            front_right
        )

        if not success:
            print(
                "Could not safely avoid obstacle. "
                "Stopping program."
            )
            break


except KeyboardInterrupt:

    print("\nInterrupted.")


finally:

    mirte.stop()

    del mirte
