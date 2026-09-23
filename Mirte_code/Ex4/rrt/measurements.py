#getting path to import KU_Mirte
import sys
import os
import math
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../Mirte/ku_mirte_python'))
from ku_mirte import KU_Mirte

#init mirte mirte
mirte = KU_Mirte()

#set driving modfier and start driving

while (True):
    msg = input('press (q) to quit, or press any key to drive forward:\n')
    if msg == 'q':
        break
    else:
        OM_lin_speed = 0.30
        #OM_ang_speed = -0.0035
        #OM_time = 2.55

        #new
        #  -0.00365
        # time:2.7
        OM_ang_speed = float(input('ang:'))
        OM_time = float(input('time:'))
        mirte.drive(OM_lin_speed, OM_ang_speed, OM_time)

while (True):
    msg = input('press (q) to quit, or key to turn:\n')
    if msg == 'q':
        break
    else:
        LT_lin_speed = 0
        #OM_ang_speed = -0.0035
        #OM_time = 2.55
        LT_ang_speed = 0.7  # rad/s

        LT_sign = float(input('sign: '))
        LT_degrees = float(input('degrees: '))
        LT_time = math.radians(LT_degrees) / LT_ang_speed
        mirte.drive(LT_lin_speed, LT_sign * LT_ang_speed, LT_time)
del mirte