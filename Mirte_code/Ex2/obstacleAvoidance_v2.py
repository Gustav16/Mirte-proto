# Drive straight ahead and detour around anything blocking the path.
#
# Path plan: drive forward in a straight line. When the front sonars see an
# obstacle inside SAFE_DISTANCE, stop and route around it with a rectangular
# 90-degree detour, then resume the original heading *and* the original line:
#
#   turn 90 (toward whichever side has more room)   -- turn away from it
#   -> drive forward                                -- clear its width
#   -> turn 90 back                                  -- facing forward again
#   -> drive forward                                -- pass alongside it
#   -> turn 90 again (same way as the turn back)     -- facing sideways again
#   -> drive forward                                -- back onto the original line
#   -> turn 90 again                                 -- facing forward, same path as before
#
# Sensor note: the assignment asks for at least 3 front-facing sonar sensors,
# but this robot's ROS wrapper (see sonar_sub.py) only exposes two front
# sonars -- front_left and front_right (plus rear_left/rear_right, which face
# the wrong way to help here). There is no third front sensor anywhere in the
# API to read from, so this script uses the two that actually exist.

import sys
import os
import math
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Mirte/ku_mirte_python'))
from ku_mirte import KU_Mirte


# -------------------------
# Tunable constants
# -------------------------

# Speed/spin values and modifiers carried over from ContinuousDrive.py, which
# calibrated them against the real robot (see that file's comments).
FWD_SPEED = 0.35   # m/s, must stay above ~0.12 or the robot will not move at all
SPIN_RATE = 1.5    # rad/s, has to be this high or the wheels sit in the dead zone
DRIFT_FIX = 0.0    # TODO: measure drift correction at FWD_SPEED, same as ContinuousDrive.py

BASE_SPEED_MOD = 2.38
BASE_TURN_MOD = 2.38

SAFE_DISTANCE = 0.35    # m, stop and detour once either front sonar reads below this
POLL_INTERVAL = 0.05    # s, how often to re-check the sonars while driving

TURN_DURATION = (math.pi / 2) / SPIN_RATE   # seconds to turn 90 degrees at SPIN_RATE

SIDESTEP_TIME = 2.0   # s spent driving sideways off / back onto the original line
BYPASS_TIME = 3.0     # s spent driving alongside the obstacle before cutting back in
# TODO: tune SIDESTEP_TIME/BYPASS_TIME on the real robot against the actual
# obstacles and robot width, so the detour reliably clears them.


# -------------------------
# Robot setup
# -------------------------

mirte = KU_Mirte()
mirte.set_driving_modifier(BASE_SPEED_MOD, BASE_TURN_MOD)


# -------------------------
# Helpers
# -------------------------

def front_clearance():
    """
    Returns (min_front_distance, front_left, front_right).
    front_left/front_right are the only front-facing sonars this robot's
    API exposes -- see the sensor note at the top of this file.
    """
    d = mirte.sonar
    fl = d['front_left']
    fr = d['front_right']
    return min(fl, fr), fl, fr


def drive_until_blocked(max_duration=None):
    """
    Drives straight forward, polling the front sonars, until either an
    obstacle closes within SAFE_DISTANCE or max_duration seconds pass
    (max_duration=None means no time limit -- only an obstacle stops it).

    Always leaves the robot stopped before returning.
    Returns (blocked: bool, front_left, front_right). front_left/front_right
    are only meaningful when blocked is True.
    """
    mirte.drive(FWD_SPEED, DRIFT_FIX, None, blocking=False)
    start = time.time()
    while max_duration is None or time.time() - start < max_duration:
        clearance, fl, fr = front_clearance()
        if clearance < SAFE_DISTANCE:
            mirte.stop()
            return True, fl, fr
        time.sleep(POLL_INTERVAL)
    mirte.stop()
    return False, None, None


def turn(direction):
    """Pivot in place 90 degrees. direction: +1 = left, -1 = right."""
    mirte.drive(0.0, direction * SPIN_RATE, TURN_DURATION, blocking=True)


def drive_leg(duration, label):
    """One forward leg of the detour. Bails out early (and says so) if a
    new obstacle appears mid-leg instead of driving the full time blind."""
    blocked, fl, fr = drive_until_blocked(max_duration=duration)
    if blocked:
        print(f"  {label}: new obstacle at L={fl:.2f}m R={fr:.2f}m, stopping early.")


def avoid_obstacle(front_left, front_right):
    """
    Rectangular detour around whatever is blocking the front sonars: turn
    toward whichever side currently has more clearance, drive around the
    obstacle, and return to the original heading and path.
    """
    # +1 = turn left first, -1 = turn right first.
    turn_dir = 1 if front_left >= front_right else -1
    side = "left" if turn_dir == 1 else "right"
    print(f"Blocked (L={front_left:.2f}m R={front_right:.2f}m); routing around to the {side}.")

    turn(turn_dir)                          # 1. turn away from the obstacle
    drive_leg(SIDESTEP_TIME, "clear width")  # 2. clear the obstacle's width
    turn(-turn_dir)                         # 3. turn back to the original heading
    drive_leg(BYPASS_TIME, "pass alongside")  # 4. drive alongside the obstacle
    turn(-turn_dir)                         # 5. turn again, same way as step 3
    drive_leg(SIDESTEP_TIME, "return to line")  # 6. return to the original line
    turn(turn_dir)                          # 7. turn back to the original heading

    print("Detour complete, back on the original path.")


# -------------------------
# Main loop: drive straight, detour around anything in the way
# -------------------------

try:
    print(f"Driving forward, avoiding obstacles closer than {SAFE_DISTANCE:.2f}m. Press Ctrl+C to stop.")
    while True:
        _, fl, fr = drive_until_blocked()  # returns only once something blocks the path
        avoid_obstacle(fl, fr)

except KeyboardInterrupt:
    print("\nInterrupted, stopping.")

finally:
    mirte.stop()
    del mirte
