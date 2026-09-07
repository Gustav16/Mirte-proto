import sys
import os
import math
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../Mirte/ku_mirte_python'))
from Mirte.ku_mirte_python.ku_mirte import KU_Mirte


mirte = KU_Mirte()

FWD_SPEED = 0.2
TURN_SPEED = 1.0
FORWARD_STEP = 0.1
TURN_TIME = math.pi / (2 * TURN_SPEED)
BYPASS_TIME = 2.0


SAFE_DISTANCE = 0.3


def read_front_sonars():
    distances = mirte.sonar
    return distances['front_left'], distances['front_right']


def choose_detour_side(front_left, front_right):
    """Return 1 for left or -1 for right, choosing the clearer side."""
    return 1 if front_left >= front_right else -1


def avoid_obstacle(turn_direction):
    """Pass the obstacle and return to the original path and heading."""
    opposite_direction = -turn_direction

    # Move around the obstacle, then make the matching turns back to the path.
    mirte.drive(0.0, turn_direction * TURN_SPEED, TURN_TIME)
    mirte.drive(FWD_SPEED, 0.0, BYPASS_TIME)
    mirte.drive(0.0, opposite_direction * TURN_SPEED, TURN_TIME)
    mirte.drive(FWD_SPEED, 0.0, BYPASS_TIME)
    mirte.drive(0.0, opposite_direction * TURN_SPEED, TURN_TIME)
    mirte.drive(FWD_SPEED, 0.0, BYPASS_TIME)
    mirte.drive(0.0, turn_direction * TURN_SPEED, TURN_TIME)


try:
    while True:
        msg = input('press (q) to quit, or press any other key to drive forward:\n')
        if msg.lower() == 'q':
            break

        while True:
            front_left, front_right = read_front_sonars()
            if front_left <= SAFE_DISTANCE or front_right <= SAFE_DISTANCE:
                turn_direction = choose_detour_side(front_left, front_right)
                side = 'left' if turn_direction == 1 else 'right'
                print(f'obstacle detected; detouring {side}')
                mirte.stop()
                avoid_obstacle(turn_direction)
                break

            mirte.drive(FWD_SPEED, 0.0, FORWARD_STEP)
finally:
    mirte.stop()
    del mirte



