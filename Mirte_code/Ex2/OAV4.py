import sys
import os
import math
import time
import select
import termios
import tty
from collections import deque

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../Mirte/ku_mirte_python'
    )
)

from ku_mirte import KU_Mirte


mirte = KU_Mirte()


# --------------------------------------------------
# Robot setup
# --------------------------------------------------

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
# Settings
# --------------------------------------------------

# Stopgrænse
SAFE_DISTANCE = 0.15

# En retning regnes først som rigtig fri over denne afstand
CLEAR_DISTANCE = 0.35

# Start midlertidig registrering af et objekt her
TRACK_DISTANCE = 1.00

STEP_TIME = 0.05
SCAN_SETTLE = 0.10

MOVE_CHUNK_STEPS = 6
RETURN_AFTER_STEPS = 6

MAX_AVOID_STEPS = 500

# Fast RAM-grænse:
# De rå samples for det aktuelle objekt kan aldrig blive større end dette.
MAX_SAMPLES_PER_OBJECT = 200

# Vi gemmer kun små summaries for de seneste objekter.
MAX_OBJECT_SUMMARIES = 20


# --------------------------------------------------
# Intern position
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


# --------------------------------------------------
# Midlertidige sensor-data
#
# ALT her ligger kun i RAM.
# Programmet skriver ikke sensor-data til disk.
# --------------------------------------------------

current_samples = deque(
    maxlen=MAX_SAMPLES_PER_OBJECT
)

object_summaries = deque(
    maxlen=MAX_OBJECT_SUMMARIES
)

current_object_active = False
current_object_id = 0
current_object_heading = 0

ping_number = 0


# --------------------------------------------------
# Q stop-knap
# --------------------------------------------------

terminal_settings = None


class UserStop(Exception):
    pass


def enable_q_key():
    global terminal_settings

    if sys.stdin.isatty():
        terminal_settings = termios.tcgetattr(
            sys.stdin.fileno()
        )
        tty.setcbreak(sys.stdin.fileno())


def restore_terminal():
    global terminal_settings

    if terminal_settings is not None:
        termios.tcsetattr(
            sys.stdin.fileno(),
            termios.TCSADRAIN,
            terminal_settings
        )
        terminal_settings = None


def q_pressed():

    if not sys.stdin.isatty():
        return False

    ready, _, _ = select.select(
        [sys.stdin],
        [],
        [],
        0
    )

    if not ready:
        return False

    key = sys.stdin.read(1)

    return key.lower() == 'q'


def clear_temporary_data():
    global current_object_active
    global current_object_id
    global current_object_heading
    global ping_number

    current_samples.clear()
    object_summaries.clear()

    current_object_active = False
    current_object_id = 0
    current_object_heading = 0
    ping_number = 0


def check_q():

    if q_pressed():

        mirte.stop()

        clear_temporary_data()

        print(
            '\nQ pressed - robot stopped '
            'and temporary sensor data cleared.'
        )

        raise UserStop


def safe_sleep(duration):

    end_time = time.time() + duration

    while time.time() < end_time:

        check_q()

        time.sleep(0.01)


# --------------------------------------------------
# Position tracking
# --------------------------------------------------

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
# Midlertidig objekt-historik
# --------------------------------------------------

def start_object():

    global current_object_active
    global current_object_id
    global current_object_heading

    current_object_id += 1
    current_object_active = True
    current_object_heading = heading

    current_samples.clear()

    print()
    print(
        f'OBJECT {current_object_id} START'
    )
    print(
        ' ping |    FL |    FR |    RL |    RR'
    )


def record_sample(distances):

    global current_object_active

    front_clearance = min(
        distances['front_left'],
        distances['front_right']
    )

    # Vi begynder først at gemme, når noget er tæt nok.
    if (
        not current_object_active
        and front_clearance <= TRACK_DISTANCE
    ):
        start_object()

    if current_object_active:

        current_samples.append(
            (
                distances['front_left'],
                distances['front_right'],
                distances['rear_left'],
                distances['rear_right']
            )
        )


def finish_object():

    global current_object_active

    if (
        not current_object_active
        or len(current_samples) == 0
    ):
        return None

    count = len(current_samples)

    avg_fl = sum(
        sample[0]
        for sample in current_samples
    ) / count

    avg_fr = sum(
        sample[1]
        for sample in current_samples
    ) / count

    avg_rl = sum(
        sample[2]
        for sample in current_samples
    ) / count

    avg_rr = sum(
        sample[3]
        for sample in current_samples
    ) / count


    # Hvis venstre front-sonar i gennemsnit så
    # kortere afstand, ligger obstaclen mest mod venstre.
    # Derfor foretrækker vi at dreje HØJRE.
    if avg_fl <= avg_fr:

        turn_direction = -1
        chosen_side = 'right'

    else:

        turn_direction = 1
        chosen_side = 'left'


    preferred_heading = (
        current_object_heading
        + turn_direction
    ) % 4


    # Gem kun et lille summary.
    # De mange rå samples bliver slettet bagefter.
    object_summaries.append(
        {
            'object': current_object_id,
            'samples': count,
            'avg_fl': avg_fl,
            'avg_fr': avg_fr,
            'avg_rl': avg_rl,
            'avg_rr': avg_rr,
            'preferred_heading': preferred_heading
        }
    )


    print(
        f'OBJECT {current_object_id} END | '
        f'samples={count}'
    )

    print(
        f'average FL={avg_fl:.2f} | '
        f'FR={avg_fr:.2f} | '
        f'RL={avg_rl:.2f} | '
        f'RR={avg_rr:.2f}'
    )

    print(
        f'Obstacle was closer on '
        f'{"left" if avg_fl <= avg_fr else "right"} '
        f'-> prefer {chosen_side}'
    )

    print()


    # Rå data slettes med det samme.
    current_samples.clear()
    current_object_active = False

    return preferred_heading


def discard_current_object():

    global current_object_active

    current_samples.clear()
    current_object_active = False


# --------------------------------------------------
# Sonar
# --------------------------------------------------

def ping_sonars(track=True):

    global ping_number

    check_q()

    distances = mirte.sonar

    ping_number += 1

    print(
        f'{ping_number:5d} | '
        f'{distances["front_left"]:5.2f} | '
        f'{distances["front_right"]:5.2f} | '
        f'{distances["rear_left"]:5.2f} | '
        f'{distances["rear_right"]:5.2f}'
    )

    if track:
        record_sample(distances)

    return distances


def obstacle_ahead(track=True):

    distances = ping_sonars(
        track=track
    )

    front_left = distances['front_left']
    front_right = distances['front_right']

    blocked = (
        front_left <= SAFE_DISTANCE
        or front_right <= SAFE_DISTANCE
    )

    return (
        blocked,
        front_left,
        front_right
    )


def front_and_rear_clearance(track=False):

    distances = ping_sonars(
        track=track
    )

    front_clearance = min(
        distances['front_left'],
        distances['front_right']
    )

    rear_clearance = min(
        distances['rear_left'],
        distances['rear_right']
    )

    return (
        front_clearance,
        rear_clearance
    )


# --------------------------------------------------
# Movement
# --------------------------------------------------

def move_one_step():

    mirte.drive(
        FWD_SPEED,
        DRIFT_FIX,
        None,
        blocking=False
    )

    safe_sleep(STEP_TIME)

    update_position()


def turn_90(direction):
    global heading

    mirte.stop()

    # Non-blocking så Q stadig kan stoppe robotten
    # midt i en drejning.
    mirte.drive(
        0.0,
        direction * SPIN_RATE,
        None,
        blocking=False
    )

    safe_sleep(TURN_DURATION)

    mirte.stop()

    heading = (
        heading + direction
    ) % 4


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


# --------------------------------------------------
# Scan alle 4 retninger
#
# Front + rear
# drej 90 grader
# front + rear igen
# --------------------------------------------------

def scan_directions():

    print('Scanning surroundings...')

    clearances = {}

    safe_sleep(SCAN_SETTLE)

    front_clearance, rear_clearance = (
        front_and_rear_clearance(
            track=False
        )
    )

    clearances[heading] = (
        front_clearance
    )

    clearances[
        (heading + 2) % 4
    ] = rear_clearance


    turn_90(1)

    safe_sleep(SCAN_SETTLE)

    front_clearance, rear_clearance = (
        front_and_rear_clearance(
            track=False
        )
    )

    clearances[heading] = (
        front_clearance
    )

    clearances[
        (heading + 2) % 4
    ] = rear_clearance


    print(
        f'forward={clearances.get(0, 0):.2f} | '
        f'left={clearances.get(1, 0):.2f} | '
        f'back={clearances.get(2, 0):.2f} | '
        f'right={clearances.get(3, 0):.2f}'
    )

    return clearances


# --------------------------------------------------
# Vælg retning
# --------------------------------------------------

def choose_direction(
    clearances,
    hit_x,
    preferred_heading=None
):

    def is_clear(direction):

        return (
            clearances.get(
                direction,
                0
            )
            > CLEAR_DISTANCE
        )


    if y_steps > 0:
        toward_line = 3

    elif y_steps < 0:
        toward_line = 1

    else:
        toward_line = None


    # Første prioritet:
    # tilbage mod original linje når vi er forbi obstaclen.
    if (
        toward_line is not None
        and x_steps >= (
            hit_x
            + RETURN_AFTER_STEPS
        )
        and is_clear(toward_line)
    ):
        return toward_line


    # Hvis et færdigt objekt har fortalt os,
    # hvilken side der gennemsnitligt var bedst,
    # bruger vi den hvis den stadig er sikker.
    if (
        preferred_heading is not None
        and is_clear(preferred_heading)
    ):
        return preferred_heading


    # Ellers foretrækker vi original fremadretning.
    if is_clear(0):
        return 0


    # Ellers vælg den frieste side.
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


    # Bagud kun som sidste udvej.
    if is_clear(2):
        return 2


    return None


# --------------------------------------------------
# Kør lille stykke
# --------------------------------------------------

def drive_chunk(direction):

    turn_to_heading(direction)

    for _ in range(
        MOVE_CHUNK_STEPS
    ):

        blocked, _, _ = (
            obstacle_ahead(
                track=True
            )
        )

        if blocked:

            mirte.stop()

            # Objektet vi har nærmet os bliver nu
            # afsluttet og giver en anbefalet retning.
            preferred_heading = (
                finish_object()
            )

            return preferred_heading


        move_one_step()


        if (
            direction in (1, 3)
            and y_steps == 0
        ):
            mirte.stop()
            return None


    mirte.stop()

    return None


# --------------------------------------------------
# Original path
# --------------------------------------------------

def original_path_found(hit_x):

    if y_steps != 0:
        return False

    if x_steps <= hit_x:
        return False

    previous_heading = heading

    turn_to_heading(0)

    blocked, _, _ = (
        obstacle_ahead(
            track=False
        )
    )

    if not blocked:
        return True

    turn_to_heading(
        previous_heading
    )

    return False


# --------------------------------------------------
# Obstacle avoidance
# --------------------------------------------------

def avoid_obstacle(
    first_preferred_heading=None
):

    hit_x = x_steps
    avoid_steps = 0

    preferred_heading = (
        first_preferred_heading
    )

    print(
        f'Obstacle detected at x={hit_x}'
    )


    while (
        avoid_steps
        < MAX_AVOID_STEPS
    ):

        check_q()


        # Hvis vi allerede er begyndt at registrere et objekt
        # foran os, fortsætter vi i samme retning og samler
        # flere samples. Vi scanner ikke/roterer ikke midt i
        # en aktiv objektmåling.
        if current_object_active:

            new_preference = (
                drive_chunk(
                    heading
                )
            )

            if (
                new_preference
                is not None
            ):
                preferred_heading = (
                    new_preference
                )

            avoid_steps += (
                MOVE_CHUNK_STEPS
            )

            continue


        if original_path_found(
            hit_x
        ):

            print(
                'Original path found again'
            )

            turn_to_heading(0)

            return True


        clearances = (
            scan_directions()
        )


        direction = (
            choose_direction(
                clearances,
                hit_x,
                preferred_heading
            )
        )


        # En preference gælder kun for
        # den næste beslutning.
        preferred_heading = None


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
            f'Choosing '
            f'{names[direction]} | '
            f'x={x_steps} | '
            f'y={y_steps}'
        )


        new_preference = (
            drive_chunk(
                direction
            )
        )


        if (
            new_preference
            is not None
        ):
            preferred_heading = (
                new_preference
            )


        avoid_steps += (
            MOVE_CHUNK_STEPS
        )


    print(
        'Maximum avoidance '
        'distance reached'
    )

    mirte.stop()

    return False


# --------------------------------------------------
# Main
# --------------------------------------------------

try:

    msg = input(
        'Press ENTER to start, '
        'or q + ENTER to quit:\n'
    )

    if msg.lower() == 'q':
        raise UserStop


    reset_tracking()
    clear_temporary_data()

    enable_q_key()


    print(
        'Driving forward. '
        'Press q at any time to stop.'
    )

    print(
        ' ping |    FL |    FR |    RL |    RR'
    )


    while True:

        blocked, _, _ = (
            obstacle_ahead(
                track=True
            )
        )


        if blocked:

            mirte.stop()


            # Gem kun summary for dette objekt.
            # Rå samples slettes i finish_object().
            preferred_heading = (
                finish_object()
            )


            success = (
                avoid_obstacle(
                    preferred_heading
                )
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
                    'a safe route'
                )


            break


        move_one_step()


except UserStop:

    pass


except KeyboardInterrupt:

    print(
        '\nKeyboard interrupt - stopping'
    )


finally:

    mirte.stop()

    clear_temporary_data()

    restore_terminal()

    del mirte
