import sys
import os
import math
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Mirte/ku_mirte_python'))
from ku_mirte import KU_Mirte


mirte = KU_Mirte()



# Kalibrerede bevægelsesværdier fra square.py


# Ligeud
FWD_SPEED = 0.35

# Lille angular correction så robotten faktisk kører lige
FWD_ANG_SPEED = -0.0035

# Drejning
TURN_SPEED = 1.0
TURN_SCALER = 0.95

# square.py bruger denne tid til ca. 90 grader
TURN_TIME = math.pi / 2



# Obstacle avoidance settings


# Ét lille step frem.
# Med den kalibrerede fart er dette et kort stykke,
# så vi kan kontrollere sonar ofte.
FORWARD_STEP = 0.1

# Stop hvis noget er tættere end 30 cm foran
SAFE_DISTANCE = 0.3

# Hvis noget på siden er tættere end dette,
# antager vi at væggen stadig er der
WALL_DISTANCE = 0.6

# Kræv 3 målinger uden væg før vi tror på,
# at kanten faktisk er fundet
CLEAR_READINGS = 3

# Kør lidt ekstra frem efter et hjørne,
# så hele robotten kommer forbi kanten
EDGE_STEPS = 3

# Sikkerhed så robotten ikke følger en væg for evigt
MAX_AVOID_STEPS = 500



# Intern position relativt til start
#
# heading:
# 0 = original retning
# 1 = venstre
# 2 = bagud
# 3 = højre


x_steps = 0
y_steps = 0
heading = 0


def reset_tracking():
    global x_steps, y_steps, heading

    x_steps = 0
    y_steps = 0
    heading = 0



# Sonar


def read_sonars():
    return mirte.sonar


def read_front_sonars():
    distances = read_sonars()

    return (
        distances['front_left'],
        distances['front_right']
    )


def obstacle_ahead():
    front_left, front_right = read_front_sonars()

    blocked = (
        front_left <= SAFE_DISTANCE
        or front_right <= SAFE_DISTANCE
    )

    return blocked, front_left, front_right



# Bevægelse


def update_position():
    global x_steps, y_steps

    # Fremad relativt til den originale kurs
    if heading == 0:
        x_steps += 1

    # Venstre relativt til den originale kurs
    elif heading == 1:
        y_steps += 1

    # Bagud relativt til den originale kurs
    elif heading == 2:
        x_steps -= 1

    # Højre relativt til den originale kurs
    elif heading == 3:
        y_steps -= 1


def move_one_step():

    # Kør kontinuerligt fremad
    mirte.drive(
        FWD_SPEED,
        FWD_ANG_SPEED,
        None,
        blocking=False
    )

    # Vent lidt før næste sonar-kontrol
    time.sleep(FORWARD_STEP)

    update_position()


def turn_90(direction):
    global heading

    # Stop fremadbevægelsen før drejning
    mirte.stop()

    # direction:
    #  1 = venstre
    # -1 = højre

    mirte.drive(
        0.0,
        direction * TURN_SCALER * TURN_SPEED,
        TURN_TIME
    )

    heading = (heading + direction) % 4


def turn_to_heading(target_heading):

    while heading != target_heading:

        difference = (target_heading - heading) % 4

        # 90 grader venstre
        if difference == 1:
            turn_90(1)

        # 90 grader højre
        elif difference == 3:
            turn_90(-1)

        # 180 grader
        else:
            turn_90(1)



# Wall following


def choose_detour_side(front_left, front_right):

    # 1 = venstre
    # -1 = højre

    if front_left >= front_right:
        return 1

    return -1


def get_wall_distance(wall_side):

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


def turn_towards_wall(wall_side):

    if wall_side == 'left':
        turn_90(1)

    else:
        turn_90(-1)


def turn_away_from_wall(wall_side):

    if wall_side == 'left':
        turn_90(-1)

    else:
        turn_90(1)



# Kan vi forlade obstaclen?


def try_leave_obstacle(hit_x):

    # Vi forlader kun obstaclen hvis:
    #
    # 1. Vi er tilbage på den originale linje
    # 2. Vi er længere fremme end det sted,
    #    hvor vi først ramte obstaclen

    if y_steps != 0 or x_steps <= hit_x:
        return False

    previous_heading = heading

    # Peg i den originale retning
    turn_to_heading(0)

    blocked, _, _ = obstacle_ahead()

    # Fri bane
    if not blocked:

        print('original path found again')
        return True

    # Der står stadig noget foran.
    # Gå tilbage til den tidligere retning
    # og fortsæt wall-following.
    print(
        'original path is still blocked; '
        'continuing along wall'
    )

    turn_to_heading(previous_heading)

    return False



# Følg obstaclens kant


def follow_obstacle(hit_x, wall_side):

    clear_count = 0
    moved_steps = 0

    while moved_steps < MAX_AVOID_STEPS:

        
        # Har vi fundet den originale linje igen?
        

        if try_leave_obstacle(hit_x):
            return True


        
        # Er der noget foran?
        

        blocked, _, _ = obstacle_ahead()

        if blocked:

            print('obstacle ahead while following wall')

            # Indvendigt hjørne / obstacle foran:
            # drej væk fra den væg vi følger
            turn_away_from_wall(wall_side)

            clear_count = 0
            continue


        
        # Er væggen stadig ved siden af?
        

        wall_distance = get_wall_distance(wall_side)

        print(
            f'wall: {wall_side} | '
            f'distance: {wall_distance:.2f} m | '
            f'x: {x_steps} | '
            f'y: {y_steps} | '
            f'heading: {heading}'
        )


        # Væggen er stadig der
        if wall_distance <= WALL_DISTANCE:

            clear_count = 0

            move_one_step()
            moved_steps += 1

            continue


        
        # Væggen ser ud til at være væk
        

        clear_count += 1


        # Vent på flere clear readings
        if clear_count < CLEAR_READINGS:

            move_one_step()
            moved_steps += 1

            continue


        
        # Muligt hjørne fundet
        

        print('corner detected')


        # Kør lidt ekstra frem, så hele robotten
        # kommer forbi hjørnet
        obstacle_during_edge = False

        for _ in range(EDGE_STEPS):

            blocked, _, _ = obstacle_ahead()

            if blocked:

                obstacle_during_edge = True
                break


            move_one_step()
            moved_steps += 1


            # Måske krydsede vi original-linjen her
            if try_leave_obstacle(hit_x):
                return True


        
        # Håndter hjørnet
        

        if obstacle_during_edge:

            # Stadig obstacle foran:
            # drej væk igen
            turn_away_from_wall(wall_side)

        else:

            # Fri bane:
            # gå rundt om hjørnet og behold
            # væggen på samme side
            turn_towards_wall(wall_side)


        clear_count = 0


    print('maximum avoidance distance reached')

    mirte.stop()

    return False



# Start obstacle avoidance


def avoid_obstacle():

    blocked, front_left, front_right = obstacle_ahead()

    if not blocked:
        return True


    # Gem hvor vi først ramte obstaclen
    hit_x = x_steps


    # Vælg siden med mest plads
    turn_direction = choose_detour_side(
        front_left,
        front_right
    )


    # Drej venstre -> obstacle på højre side
    if turn_direction == 1:

        side = 'left'
        wall_side = 'right'


    # Drej højre -> obstacle på venstre side
    else:

        side = 'right'
        wall_side = 'left'


    print(
        f'obstacle detected at x={hit_x}; '
        f'detouring {side}'
    )


    # Drej væk fra obstaclen
    turn_90(turn_direction)


    # Følg kanten indtil den originale linje
    # findes igen på den anden side
    return follow_obstacle(
        hit_x,
        wall_side
    )



# Main


try:

    while True:

        msg = input(
            'press (q) to quit, '
            'or press any other key to drive forward:\n'
        )


        if msg.lower() == 'q':
            break


        reset_tracking()

        print('driving forward')


        # Kør ligeud indtil vi møder noget
        while True:

            blocked, _, _ = obstacle_ahead()


            if blocked:

                mirte.stop()

                success = avoid_obstacle()

                mirte.stop()


                if success:

                    print(
                        'obstacle handled; '
                        'back on original path'
                    )

                else:

                    print(
                        'could not find '
                        'the original path safely'
                    )


                # Tilbage til start-menuen
                break


            # Fri bane
            move_one_step()


except KeyboardInterrupt:

    print('\nstopping')


finally:

    mirte.stop()
    del mirte