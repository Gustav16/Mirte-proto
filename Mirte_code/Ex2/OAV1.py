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

# Kalibrerede værdier fra obstacleAvoidance_v2.py
FWD_SPEED = 0.35
DRIFT_FIX = 0.0

SPIN_RATE = 1.5
BASE_SPEED_MOD = 2.38
BASE_TURN_MOD = 2.38

mirte.set_driving_modifier(
    BASE_SPEED_MOD,
    BASE_TURN_MOD
)

TURN_DURATION = (math.pi / 2) / SPIN_RATE


# --------------------------------------------------
# Obstacle avoidance settings
# --------------------------------------------------

SAFE_DISTANCE = 0.35

# Hvor ofte vi checker sonar / opdaterer vores simple position
STEP_TIME = 0.05

# Hvor tæt væggen skal være for at vi stadig følger den
WALL_DISTANCE = 0.6

# Flere målinger uden væg før vi accepterer et hjørne
CLEAR_READINGS = 5

# Ca. 0.30 sekund ekstra frem efter et hjørne
EDGE_STEPS = 6

# Sikkerhedsgrænse
MAX_AVOID_STEPS = 1000


# --------------------------------------------------
# Simpel intern position
#
# heading:
# 0 = original retning
# 1 = venstre
# 2 = bagud
# 3 = højre
# --------------------------------------------------

x_steps = 0
y_steps = 0
heading = 0


def reset_tracking():
    global x_steps, y_steps, heading

    x_steps = 0
    y_steps = 0
    heading = 0


def update_position():
    global x_steps, y_steps

    if heading == 0:
        x_steps += 1

    elif heading == 1:
        y_steps += 1

    elif heading == 2:
        x_steps -= 1

    elif heading == 3:
        y_steps -= 1


# --------------------------------------------------
# Sonar
# --------------------------------------------------

def read_sonars():
    return mirte.sonar


def obstacle_ahead():
    distances = read_sonars()

    front_left = distances['front_left']
    front_right = distances['front_right']

    blocked = (
        front_left <= SAFE_DISTANCE
        or front_right <= SAFE_DISTANCE
    )

    return blocked, front_left, front_right


def wall_distance(wall_side):
    distances = read_sonars()

    if wall_side == 'left':
        return min(
            distances['front_left'],
            distances['rear_left']
        )

    return min(
        distances['front_right'],
        distances['rear_right']
    )


# --------------------------------------------------
# Movement
# --------------------------------------------------

def move_one_step():

    # Kontinuerlig fremadkørsel.
    # STEP_TIME bestemmer kun hvor ofte vi checker sensorerne.
    mirte.drive(
        FWD_SPEED,
        DRIFT_FIX,
        None,
        blocking=False
    )

    time.sleep(STEP_TIME)

    update_position()


def turn_90(direction):
    global heading

    # direction:
    #  1 = venstre
    # -1 = højre

    mirte.stop()

    # Behold den stærke drejning fra obstacleAvoidance_v2.py
    mirte.drive(
        0.0,
        direction * SPIN_RATE,
        TURN_DURATION,
        blocking=True
    )

    heading = (heading + direction) % 4


def turn_to_heading(target_heading):

    while heading != target_heading:

        difference = (target_heading - heading) % 4

        if difference == 1:
            turn_90(1)

        elif difference == 3:
            turn_90(-1)

        else:
            turn_90(1)


# --------------------------------------------------
# Tilbage til den originale rute
# --------------------------------------------------

def try_leave_obstacle(hit_x):

    # Vi skal være:
    # 1. tilbage på den originale linje
    # 2. længere fremme end hvor obstaclen blev fundet

    if y_steps != 0 or x_steps <= hit_x:
        return False

    previous_heading = heading

    turn_to_heading(0)

    blocked, _, _ = obstacle_ahead()

    if not blocked:
        print('Original path found again')
        return True

    # Originalruten er stadig blokeret.
    # Drej tilbage og fortsæt langs obstaclen.
    turn_to_heading(previous_heading)

    return False


# --------------------------------------------------
# Wall following
# --------------------------------------------------

def follow_wall(hit_x, wall_side):

    clear_count = 0
    moved_steps = 0

    while moved_steps < MAX_AVOID_STEPS:

        # Er vi tilbage på vores originale rute?
        if try_leave_obstacle(hit_x):
            return True

        blocked, _, _ = obstacle_ahead()

        # Noget direkte foran:
        # drej væk fra den væg vi følger.
        if blocked:

            if wall_side == 'left':
                turn_90(-1)
            else:
                turn_90(1)

            clear_count = 0
            continue

        distance = wall_distance(wall_side)

        # Væggen er stadig ved siden af os.
        if distance <= WALL_DISTANCE:

            clear_count = 0
            move_one_step()
            moved_steps += 1
            continue

        # Væggen ser ud til at være væk.
        clear_count += 1

        if clear_count < CLEAR_READINGS:

            move_one_step()
            moved_steps += 1
            continue

        print('Corner detected')

        # Kør lidt frem så hele robotten kommer forbi hjørnet.
        for _ in range(EDGE_STEPS):

            blocked, _, _ = obstacle_ahead()

            if blocked:
                break

            move_one_step()
            moved_steps += 1

            if try_leave_obstacle(hit_x):
                return True

        # Hvis der nu er frit foran os,
        # drejer vi rundt om hjørnet så væggen
        # bliver på samme side igen.
        blocked, _, _ = obstacle_ahead()

        if blocked:

            if wall_side == 'left':
                turn_90(-1)
            else:
                turn_90(1)

        else:

            if wall_side == 'left':
                turn_90(1)
            else:
                turn_90(-1)

        clear_count = 0

    print('Maximum avoidance distance reached')
    mirte.stop()

    return False


# --------------------------------------------------
# Start obstacle avoidance
# --------------------------------------------------

def avoid_obstacle(front_left, front_right):

    hit_x = x_steps

    # Vælg den side med mest fri plads.
    # +1 = venstre
    # -1 = højre
    turn_direction = (
        1 if front_left >= front_right else -1
    )

    if turn_direction == 1:
        side = 'left'
        wall_side = 'right'

    else:
        side = 'right'
        wall_side = 'left'

    print(
        f'Obstacle detected at x={hit_x} | '
        f'L={front_left:.2f} m | '
        f'R={front_right:.2f} m | '
        f'detouring {side}'
    )

    # Første drej væk fra obstaclen.
    turn_90(turn_direction)

    # Følg derefter kanten indtil vi finder
    # den originale rute igen.
    return follow_wall(
        hit_x,
        wall_side
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

try:

    while True:

        msg = input(
            'Press (q) to quit, '
            'or press any other key to drive forward:\n'
        )

        if msg.lower() == 'q':
            break

        reset_tracking()

        print('Driving forward')

        while True:

            blocked, front_left, front_right = obstacle_ahead()

            if blocked:

                mirte.stop()

                success = avoid_obstacle(
                    front_left,
                    front_right
                )

                mirte.stop()

                if success:
                    print(
                        'Obstacle handled; '
                        'back on original path'
                    )

                else:
                    print(
                        'Could not find '
                        'the original path safely'
                    )

                break

            move_one_step()


except KeyboardInterrupt:

    print('\nStopping')


finally:

    mirte.stop()
    del mirte
