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
    sleep(0.4)

OM_lin_speed = 0.5
OM_ang_speed = 0
OM_time = 2

LT_lin_speed = 0.0
LT_ang_speed = 0.5
LT_time = math.pi

#non-blocking loop for driving 1 sec
while (True):
    msg = input('press (q) to quit, or press any other key to drive in a square:\n')
    if msg == 'q':
        break
    else:
        for _ in range(4):
            #drive one meter
            mirte.drive(OM_lin_speed:, OM_ang_speed, OM_time)
            
            #wait before excutiong next command
            wait()
            #turn 90 degrees left
            mirte.drive(LT_lin_speed:, LT_ang_speed, LT_time)
            wait()

# ... END ...

del mirte