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

mirte = KU_Mirte()

# Behold de værdier som virker godt på robotten
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

SAFE_DISTANCE = 0.25

# En retning skal have lidt mere plads,
# før vi aktivt vælger at køre den vej
CLEAR_DISTANCE = 0.35

# Smooth kørsel: vi stopper ikke mellem disse checks
STEP_TIME = 0.05

# Vent kort efter et drej før sonar aflæses
SCAN_SETTLE = 0.10

# Når vi undviger, kører vi lidt ad gangen og scanner igen
MOVE_CHUNK_STEPS = 6

# Vi prøver først at vende tilbage til den originale linje,
# når vi er kommet lidt længere frem end obstaclen
RETURN_AFTER_STEPS = 6

# Sikkerhed
MAX_AVOID_STEPS = 500


# --------------------------------------------------
# Simpel intern position
#
# heading:
# 0 = original retning
# 1 = venstre
# 2 = bagud
# 3 = højre
#
# y = 0 er den originale linje
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

def read_front():
    distances = mirte.sonar

    front_left = distances['front_left']
    front_right = distances['front_right']

    clearance = min(
        front_left,
        front_right
    )

    return clearance, front_left, front_right


def read_front_and_rear():
    distances = mirte.sonar

    front_clearance = min(
        distances['front_left'],
        distances['front_right']
    )

    rear_clearance = min(
        distances['rear_left'],
        distances['rear_right']
    )

    return front_clearance, rear_clearance


def obstacle_ahead():
    clearance, front_left, front_right = read_front()

    return (
        clearance <= SAFE_DISTANCE,
        front_left,
        front_right
    )

def move_one_step():

    # Kontinuerlig / smooth fremadkørsel
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

    # +1 = venstre
    # -1 = højre

    mirte.stop()

    mirte.drive(
        0.0,
        direction * SPIN_RATE,
        TURN_DURATION,
        blocking=True
    )

    heading = (heading + direction) % 4


def turn_to_heading(target_heading):

    while heading != target_heading:

        difference = (
            target_heading - heading
        ) % 4

        if difference == 1:
            turn_90(1)

        elif difference == 3:
            turn_90(-1)

        else:
            turn_90(1)

def scan_directions():

    print('Scanning surroundings...')

    clearances = {}

    # Første måling:
    # front-sonar ser den retning robotten peger
    # rear-sonar ser den modsatte retning
    time.sleep(SCAN_SETTLE)

    front_clearance, rear_clearance = read_front_and_rear()

    clearances[heading] = front_clearance
    clearances[(heading + 2) % 4] = rear_clearance


    # Drej kun 90 grader.
    # Nu kan front + rear se de to resterende retninger.
    turn_90(1)

    time.sleep(SCAN_SETTLE)

    front_clearance, rear_clearance = read_front_and_rear()

    clearances[heading] = front_clearance
    clearances[(heading + 2) % 4] = rear_clearance


    print(
        f"forward={clearances.get(0, 0):.2f} | "
        f"left={clearances.get(1, 0):.2f} | "
        f"back={clearances.get(2, 0):.2f} | "
        f"right={clearances.get(3, 0):.2f}"
    )

    return clearances

def choose_direction(clearances, hit_x):

    def is_clear(direction):
        return (
            clearances.get(direction, 0)
            > CLEAR_DISTANCE
        )

    # Hvis vi er væk fra den originale linje,
    # prøver vi at komme tilbage mod y = 0.
    if y_steps > 0:
        toward_line = 3      # højre

    elif y_steps < 0:
        toward_line = 1      # venstre

    else:
        toward_line = None


    # Hvis vi er kommet forbi obstaclen,
    # har det højeste prioritet at komme tilbage
    # på den originale linje.
    if (
        toward_line is not None
        and x_steps >= hit_x + RETURN_AFTER_STEPS
        and is_clear(toward_line)
    ):
        return toward_line


    # Derefter foretrækker vi altid den originale
    # fremadretning, hvis den er fri.
    if is_clear(0):
        return 0


    # Hvis vi står på original-linjen og fremad er blokeret,
    # vælger vi den frieste side.
    side_directions = [
        direction
        for direction in (1, 3)
        if is_clear(direction)
    ]

    if side_directions:
        return max(
            side_directions,
            key=lambda direction:
                clearances[direction]
        )


    # Bagud er kun sidste udvej.
    if is_clear(2):
        return 2


    return None

def drive_chunk(direction):

    turn_to_heading(direction)

    for _ in range(MOVE_CHUNK_STEPS):

        blocked, _, _ = obstacle_ahead()

        if blocked:
            mirte.stop()
            return

        move_one_step()

        # Hvis vi er på vej tilbage mod original-linjen,
        # stopper vi præcis når vores interne y rammer 0.
        if direction in (1, 3) and y_steps == 0:
            mirte.stop()
            return

    mirte.stop()

def original_path_found(hit_x):

    if y_steps != 0:
        return False

    if x_steps <= hit_x:
        return False

    previous_heading = heading

    turn_to_heading(0)

    blocked, _, _ = obstacle_ahead()

    if not blocked:
        return True

    # Stadig blokeret:
    # tilbage til retningen vi havde før testen
    turn_to_heading(previous_heading)

    return False

def avoid_obstacle():

    hit_x = x_steps
    avoid_steps = 0

    print(
        f'Obstacle detected at x={hit_x}'
    )

    while avoid_steps < MAX_AVOID_STEPS:

        # Er vi tilbage på den originale linje
        # og kan vi fortsætte frem?
        if original_path_found(hit_x):

            print(
                'Original path found again'
            )

            turn_to_heading(0)

            return True


        # Stop og kig hele vejen rundt
        clearances = scan_directions()

        direction = choose_direction(
            clearances,
            hit_x
        )


        # Ingen sikker retning
        if direction is None:

            print(
                'No clear direction found'
            )

            mirte.stop()
            return False


        names = {
            0: 'forward',
            1: 'left',
            2: 'back',
            3: 'right'
        }

        print(
            f'Choosing {names[direction]} | '
            f'x={x_steps} | '
            f'y={y_steps}'
        )


        drive_chunk(direction)

        avoid_steps += MOVE_CHUNK_STEPS


    print(
        'Maximum avoidance distance reached'
    )

    mirte.stop()

    return False

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

            blocked, _, _ = obstacle_ahead()

            if blocked:

                mirte.stop()

                success = avoid_obstacle()

                mirte.stop()


                if success:

                    print(
                        'Obstacle handled; '
                        'back on original path'
                    )

                else:

                    print(
                        'Could not find '
                        'a safe route'
                    )

                break


            move_one_step()


except KeyboardInterrupt:

    print('\nStopping')


finally:

    mirte.stop()
    del mirte
