#getting path to import KU_Mirte
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '../Mirte/ku_mirte_python'))

import time
from time import sleep
from ku_mirte import KU_Mirte

#init mirte mirte
mirte = KU_Mirte()

def wait():
    "Waiting function for inbetween commands"
    sleep(0.1)

CIRCLE_lin_speed = 0.3
CIRCLE_ang_speed = 0.5
CIRCLE_DURATION = 2*math.pi/CIRCLE_ang_speed #360 degrees will take 2*PI/ang_speed time

#keep doing cickles
mirte.drive(CIRCLE_lin_speed, CIRCLE_ang_speed, CIRCLE_DURATION)
while (True):
    msg = input('press (q) to quit, or press any other key to drive in a 8 figure:\n')
    if msg == 'q':
        break
    else:
        for _ in range(4):
            mirte.drive(CIRCLE_lin_speed, CIRCLE_ang_speed, CIRCLE_DURATION)

            wait()
            mirte.drive(CIRCLE_lin_speed, -CIRCLE_ang_speed, CIRCLE_DURATION)
            wait()

# ... END ...

mirte.close()