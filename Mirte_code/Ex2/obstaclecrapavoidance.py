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

mirte = KU_Mirte()

BASE_SPEED_MOD = 2.38
BASE_TURN_MOD = 2.38

mirte.set_driving_modifier(
    BASE_SPEED_MOD,
    BASE_TURN_MOD
)


# Normal forward speed
FWD_SPEED = 0.35

# Slow down when we get near an obstacle,
# but keep driving until we are quite close.
APPROACH_SPEED = 0.20

# Mecanum sideways speed
SIDE_SPEED = 0.25

# No normal rotation correction for now
FWD_ANG_SPEED = 0.0

# Start slowing down here, but DO NOT stop yet
SLOW_DISTANCE = 0.40

# Stop and start the avoidance manoeuvre only when
# we are quite close to the obstacle
STOP_DISTANCE = 0.25

# Emergency distance used while doing the avoidance manoeuvre
EMERGENCY_DISTANCE = 0.12

# Obstacle must be this far away before we consider
# the front clear after moving sideways
FRONT_CLEAR_DISTANCE = 0.42

# Number of consecutive clear measurements required
CLEAR_READINGS_REQUIRED = 4

POLL_INTERVAL = 0.05

# Maximum amount of time we allow the robot to strafe
# sideways while trying to clear the obstacle.
MAX_SIDE_TIME = 4.0

# Once the obstacle disappears from the front sonars,
# continue sideways a little further to create margin.
EXTRA_SIDE_TIME = 0.25

# Returning for the same time should approximately bring
# us back to the original line.
# Tune this after testing on the real robot if necessary.
RETURN_SCALE = 1.0

# Ignore rear detection during the first part of the forward bypass
BYPASS_MIN_TIME = 0.80

# If rear sonar never gives us a useful signal,
# use this as the normal fallback duration.
BYPASS_FALLBACK_TIME = 3.0

# Absolute maximum bypass duration
BYPASS_MAX_TIME = 4.5

# We consider the obstacle "seen behind us" when the relevant
# rear sonar becomes smaller than this value.
REAR_SEEN_DISTANCE = 0.65

# After having seen it, it must become larger than this value
# for several readings before we consider the obstacle passed.
REAR_CLEAR_DISTANCE = 0.50

REAR_CLEAR_READINGS_REQUIRED = 4

def read_sonars():
    """
    Read all four sonar sensors.

    Returns a dictionary with:
        front_left
        front_right
        rear_left
        rear_right
        front_clearance
        rear_clearance
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

def drive_until_blocked():
    """
    Drive forward continuously.

    Far from an obstacle:
        drive at FWD_SPEED

    Near an obstacle:
        slow down, but keep driving

    Only stop when the front sonar gets below STOP_DISTANCE.

    Returns the sonar readings measured when we stop.
    """

    current_speed = None

    while True:

        s = read_sonars()
        distance = s["front_clearance"]

        print_sonars("Forward", s)

        # Only stop when quite close
        if distance <= STOP_DISTANCE:

            mirte.stop()

            print(
                f"Obstacle close: {distance:.2f} m. "
                "Starting avoidance."
            )

            return s

        # Close enough that we should slow down,
        # but not stop yet
        if distance <= SLOW_DISTANCE:

            wanted_speed = APPROACH_SPEED

        else:

            wanted_speed = FWD_SPEED

        # Only send a new drive command when the wanted speed changes
        if wanted_speed != current_speed:

            mirte.drive(
                wanted_speed,
                FWD_ANG_SPEED,
                None,
                blocking=False
            )

            current_speed = wanted_speed

        time.sleep(POLL_INTERVAL)

def choose_side(s):
    """
    Choose the side with the most apparent room.

    We use BOTH the front and rear sonar on each side:

        left_score  = min(front_left, rear_left)
        right_score = min(front_right, rear_right)

    This does not give perfect sideways sensing, because the sonars
    are not true side-facing sensors, but it uses both corners of the
    robot instead of relying only on the two front measurements.

    Returns:
        +1 for left
        -1 for right
    """

    left_score = min(
        s["front_left"],
        s["rear_left"]
    )

    right_score = min(
        s["front_right"],
        s["rear_right"]
    )

    print(
        f"Side scores | "
        f"left={left_score:.2f} "
        f"right={right_score:.2f}"
    )

    if left_score >= right_score:
        return 1

    return -1

def strafe_until_clear(direction):
    """
    Move sideways with the Mecanum wheels until the obstacle
    is no longer visible in front.

    direction:
        +1 = left
        -1 = right

    Returns:
        total sideways time,
        or None if the manoeuvre fails.
    """

    side = "left" if direction > 0 else "right"

    print(f"Strafing {side}.")

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

        # Emergency check using the sensors we have.
        #
        # Note: the robot does not have true side-facing sonar,
        # so front/rear sonar cannot guarantee that the side itself
        # is completely clear.
        if (
            s["front_clearance"] <= EMERGENCY_DISTANCE
            or s["rear_clearance"] <= EMERGENCY_DISTANCE
        ):
            mirte.stop()

            print(
                "Emergency distance reached while strafing."
            )

            return None

        # Front obstacle has disappeared
        if s["front_clearance"] >= FRONT_CLEAR_DISTANCE:
            clear_readings += 1
        else:
            clear_readings = 0

        if clear_readings >= CLEAR_READINGS_REQUIRED:

            print("Front is clear.")

            # Keep going slightly further sideways
            time.sleep(EXTRA_SIDE_TIME)

            mirte.stop()

            side_time = time.monotonic() - start

            print(
                f"Sideways movement took "
                f"{side_time:.2f} s."
            )

            return side_time

        if elapsed >= MAX_SIDE_TIME:

            mirte.stop()

            print(
                "Could not clear the obstacle "
                "within MAX_SIDE_TIME."
            )

            return None

        time.sleep(POLL_INTERVAL)

def drive_past_obstacle(direction):
    """
    Drive forward beside the obstacle.

    The obstacle is on the opposite side from the direction
    we strafed:

        strafe left  -> obstacle is on our right
        strafe right -> obstacle is on our left

    Therefore:
        strafe left  -> watch rear_right
        strafe right -> watch rear_left

    We use the rear sonar as confirmation that the obstacle
    has moved behind the robot.

    If the rear sonar never sees it reliably, we fall back to
    BYPASS_FALLBACK_TIME.
    """

    if direction > 0:
        rear_key = "rear_right"
        obstacle_side = "right"
    else:
        rear_key = "rear_left"
        obstacle_side = "left"

    print(
        f"Driving forward past obstacle. "
        f"Watching {rear_key}."
    )

    start = time.monotonic()

    rear_has_seen_obstacle = False
    rear_clear_readings = 0

    mirte.drive(
        FWD_SPEED,
        FWD_ANG_SPEED,
        None,
        blocking=False
    )

    while True:

        s = read_sonars()
        elapsed = time.monotonic() - start

        print_sonars("Bypass", s)

        # A new obstacle directly ahead
        if s["front_clearance"] <= STOP_DISTANCE:

            mirte.stop()

            print(
                "New obstacle detected in front "
                "during bypass."
            )

            return False

        rear_value = s[rear_key]

        # Do not trust rear detection immediately after starting
        if elapsed >= BYPASS_MIN_TIME:

            if rear_value <= REAR_SEEN_DISTANCE:
                rear_has_seen_obstacle = True
                rear_clear_readings = 0

            elif (
                rear_has_seen_obstacle
                and rear_value >= REAR_CLEAR_DISTANCE
            ):
                rear_clear_readings += 1

            elif rear_has_seen_obstacle:
                rear_clear_readings = 0

        # Rear sonar saw the obstacle and now sees free space again.
        # This is our preferred signal that we have passed it.
        if (
            rear_has_seen_obstacle
            and rear_clear_readings
            >= REAR_CLEAR_READINGS_REQUIRED
        ):

            mirte.stop()

            print(
                f"Obstacle passed according to "
                f"{rear_key} on the {obstacle_side} side."
            )

            return True

        # Rear sonar did not give us a useful signal.
        # Use the normal timed fallback instead.
        if (
            not rear_has_seen_obstacle
            and elapsed >= BYPASS_FALLBACK_TIME
        ):

            mirte.stop()

            print(
                "Rear sonar did not clearly detect the obstacle. "
                "Using timed bypass fallback."
            )

            return True

        # Hard safety limit
        if elapsed >= BYPASS_MAX_TIME:

            mirte.stop()

            print(
                "Maximum bypass time reached."
            )

            return True

        time.sleep(POLL_INTERVAL)

def return_to_line(direction, side_time):
    """
    Move sideways in the opposite direction for approximately
    the same time as the initial sidestep.

    Front AND rear sonar are checked while returning.
    """

    return_direction = -direction
    return_time = side_time * RETURN_SCALE

    side = "left" if return_direction > 0 else "right"

    print(
        f"Returning {side} toward original line "
        f"for {return_time:.2f} s."
    )

    start = time.monotonic()

    mirte.drive(
        [0.0, return_direction * SIDE_SPEED],
        0.0,
        None,
        blocking=False
    )

    while time.monotonic() - start < return_time:

        s = read_sonars()

        print_sonars("Return", s)

        # If either front or rear becomes dangerously close,
        # stop rather than forcing the return.
        if (
            s["front_clearance"] <= EMERGENCY_DISTANCE
            or s["rear_clearance"] <= EMERGENCY_DISTANCE
        ):
            mirte.stop()

            print(
                "Emergency distance reached while "
                "returning to original line."
            )

            return False

        time.sleep(POLL_INTERVAL)

    mirte.stop()

    print(
        "Back approximately on the original line."
    )

    return True

def avoid_obstacle(initial_sonars):
    """
    1. Use front + rear sonar to choose the better side.
    2. Strafe sideways until the front is clear.
    3. Drive forward past the obstacle.
       Rear sonar helps confirm when it is behind us.
    4. Strafe back for the same approximate amount of time.
    """

    direction = choose_side(initial_sonars)

    side = "left" if direction > 0 else "right"

    print(f"Choosing {side}.")

    # Step 1: move sideways
    side_time = strafe_until_clear(direction)

    if side_time is None:

        mirte.stop()

        print(
            "Could not safely clear obstacle sideways."
        )

        return False

    # Step 2: move forward past it
    if not drive_past_obstacle(direction):

        mirte.stop()

        print(
            "Could not safely pass obstacle."
        )

        return False

    # Step 3: return toward original line
    if not return_to_line(
        direction,
        side_time
    ):

        mirte.stop()

        print(
            "Could not safely return to original line."
        )

        return False

    print(
        "Avoidance complete. "
        "Continuing forward.\n"
    )

    return True

try:

    print(
        "Driving forward with Mecanum obstacle avoidance."
    )

    print(
        f"Slow down at {SLOW_DISTANCE:.2f} m, "
        f"but only stop at {STOP_DISTANCE:.2f} m."
    )

    print(
        "Front and rear sonar are used during avoidance."
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    while True:

        obstacle_sonars = drive_until_blocked()

        success = avoid_obstacle(
            obstacle_sonars
        )

        if not success:

            print(
                "Avoidance failed. "
                "Robot remains stopped."
            )

            break


except KeyboardInterrupt:

    print("\nInterrupted.")


finally:

    mirte.stop()

    del mirte
