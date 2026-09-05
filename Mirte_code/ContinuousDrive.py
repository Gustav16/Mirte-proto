# Libraries.
#getting path to import KU_Mirte
import sys
import os
import math
import time
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '../Mirte/ku_mirte_python'))
from ku_mirte import KU_Mirte

# --- Setup ---
blocked = False
rear_blocked = False
STOP_DIST = 0.3 # Distance threshold in meters to stop the robot
running = True

# What the robot is currently commanded to do, so the watchdog knows which
# sensor readings actually matter. A spin does not translate the robot, so a
# wall in front of us is not a reason to interrupt it.
motion = "stopped"  # "forward" | "spin" | "reverse" | "stopped"

# --- Speed constants ---
# Linear speed is in m/s, angular speed is in rad/s. Positive angular is left.
# Turn radius is lin_speed / ang_speed, so FWD_SPEED with TURN_RATE gives 0.4m.
FWD_SPEED = 0.15   # must stay above ~0.12 or the robot will not move at all
REV_SPEED = -0.15
TURN_RATE = 0.375  # scaled with FWD_SPEED so the arc radius stays at 0.4m
SPIN_RATE = 0.4    # for turning on the spot, no forward motion

# Drift correction. This is the angular speed we have to add to make the robot
# actually drive straight when we ask it to go straight.
# TODO: measure this at FWD_SPEED. The -0.003625 we found in test.py was measured
# at 0.35 m/s, so it most likely does not transfer directly to 0.15 m/s.
DRIFT_FIX = 0.0
# TODO: check if reverse drifts the same way as forward. If it does not, this
# needs its own measured value instead of reusing DRIFT_FIX.
REV_DRIFT_FIX = DRIFT_FIX
# TODO: decide if the arcs need drift correction too. It matters much less there
# than when driving straight, so it is left off for now.

#init mirte mirte
mirte = KU_Mirte()

# ground check, we will continuesly check if something is infront of the
# robot, and print the distance to the object infront of the robot..
# this is optional and needs a argument to be passed to the script, if you want to use it.
# follow python3 ContinuousDrive.py --groundcheck
# This is a measurement mode, the script exits when you stop it.

def ground_check():
    print("Ground check started. Press Ctrl+C to stop.")
    try:
        while True:
            d = mirte.sonar
            front = min(d['front_left'], d['front_right'])
            print(f"Distance to object in front: {front:.2f}m")
            time.sleep(1)  # Check every second
    except KeyboardInterrupt:
        print("Ground check stopped.")

# Run this before the watchdog starts, so it does not print blocked messages
# at us while we are trying to read distances.
if "--groundcheck" in sys.argv:
    ground_check()
    del mirte
    sys.exit(0)

#Before creating the driving loop, we need a safety mehchnism that avoids the
# robot to drive into a wall. This is done by using the distance sensor.


# We will initiate a get sonar loop, that continually monitors and.
# stops the robot if it is too close to a wall.
def sonar_watchdog():
    global blocked, rear_blocked, motion
    while running:
        d = mirte.sonar
        front = min(d['front_left'], d['front_right'])
        rear = min(d['rear_left'], d['rear_right'])

        # blocked just tracks whether something is in front of us. Whether we
        # actually stop depends on if we are driving into it or not.
        if front < STOP_DIST:
            if motion == "forward" and not blocked:
                mirte.stop()
                motion = "stopped"
                print(f"blocked in front at {front:.2f}m")
            blocked = True
        else:
            blocked = False

        if rear < STOP_DIST:
            if motion == "reverse" and not rear_blocked:
                mirte.stop()
                motion = "stopped"
                print(f"blocked behind at {rear:.2f}m")
            rear_blocked = True
        else:
            rear_blocked = False

        time.sleep(0.05)  # Sleep for 50 milliseconds to avoid busy waiting

t = threading.Thread(target=sonar_watchdog, daemon=True)
t.start()

# --- Main driving loop ---
# Every command latches, the robot keeps doing it until we send something else.
# We record what kind of motion we asked for in `motion`, so the watchdog knows
# which sonar readings are relevant. Forward and the arcs are guarded by the
# front sensors, reverse by the rear ones, and the spins are not guarded at all
# since they do not move the robot anywhere.
print(f"arc radius is {FWD_SPEED / TURN_RATE:.2f}m at the current speeds")

try:
    while True:
        key = input("cmd (w=fwd, a/d=arc, z/c=spin, x=rev, s=stop, q=quit): ").strip().lower()

        if key == 'q':
            break  # Exit the loop and stop the program
        elif key == 'w':
            if blocked:
                print("Cannot drive forward, obstacle detected!")
                continue  # Skip driving forward if blocked
            motion = "forward"
            mirte.drive(FWD_SPEED, DRIFT_FIX, None, blocking = False)  # Drive forward
        elif key == 'a':
            if blocked:
                print("Cannot drive forward, obstacle detected!")
                continue
            motion = "forward"
            mirte.drive(FWD_SPEED, TURN_RATE, None, blocking = False)  # Arc to the left
        elif key == 'd':
            if blocked:
                print("Cannot drive forward, obstacle detected!")
                continue
            motion = "forward"
            mirte.drive(FWD_SPEED, -TURN_RATE, None, blocking = False)  # Arc to the right
        elif key == 'z':
            motion = "spin"
            mirte.drive(0.0, SPIN_RATE, None, blocking = False)  # Spin on the spot, left
        elif key == 'c':
            motion = "spin"
            mirte.drive(0.0, -SPIN_RATE, None, blocking = False)  # Spin on the spot, right
        elif key == 'x':
            if rear_blocked:
                print("Cannot reverse, obstacle behind!")
                continue
            motion = "reverse"
            mirte.drive(REV_SPEED, REV_DRIFT_FIX, None, blocking = False)  # Reverse
        elif key == 's':
            mirte.stop()  # Stop the robot
            motion = "stopped"
        else:
            print(f"Unknown command: {key}")
except KeyboardInterrupt:
    # Without this the robot keeps the last velocity latched after we ctrl+c out.
    print("\nInterrupted, stopping.")

running = False  # Stop the sonar watchdog thread
t.join()  # Wait for it to exit before we delete mirte out from under it
mirte.stop()  # Ensure the robot is stopped before exiting
del mirte  # Clean up the mirte object
