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

OM_lin_speed = 0.35
OM_ang_speed = -0.0035
OM_time = 2.55

LT_lin_speed = 0.0
LT_ang_speed = 1
LT_scaler = 0.95
LT_time = math.pi/2

#non-blocking loop for driving 1 sec
while (True):
    msg = input('press (q) to quit, or press any other key to drive in a square:\n')
    if msg == 'q':
        break
    else:
        LT_scaler = float(input('left turn scaler:'))
        for _ in range(4):
            #drive one meter
            mirte.drive(OM_lin_speed, OM_ang_speed, OM_time)

            time.sleep(0.4)
            
            #turn 90 degrees left
            mirte.drive(LT_lin_speed, LT_scaler*LT_ang_speed, LT_time)
            time.sleep(0.4)