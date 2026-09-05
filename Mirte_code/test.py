#getting path to import KU_Mirte
import sys
import os
import math
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../Mirte/ku_mirte_python'))
from ku_mirte import KU_Mirte

#init mirte mirte
mirte = KU_Mirte()

#set driving modfier and start driving

while (True):
    msg = input('press (q) to quit, or press any other key to drive in a square:\n')
    if msg == 'q':
        break
    else:
        OM_lin_speed = 0.35
        #OM_ang_speed = -0.0035
        #OM_time = 2.55

        #new
        #  -0.00365
        # time:2.7
        OM_ang_speed = float(input('ang:'))
        OM_time = float(input('time:'))
        mirte.drive(OM_lin_speed, OM_ang_speed, OM_time)

while (True):
    msg = input('press (q) to quit, or press any other key rotate:\n')
    if msg == 'q':
        break
    else:
        LT_lin_speed = 0
        #OM_ang_speed = -0.0035
        #OM_time = 2.55
        LT_ang_speed = 1
        LT_scaler = float(input('scaler:'))
        LT_time = math.pi/2
        mirte.drive(LT_lin_speed, LT_scaler*LT_ang_speed, LT_time)

OM_lin_speed = 0.35
OM_ang_speed = -0.00362
OM_time = 2.7

LT_lin_speed = 0
#OM_ang_speed = -0.0035
#OM_time = 2.55
LT_ang_speed = 1
LT_scaler = 0.927
LT_time = math.pi/2

for i in range(4):
    mirte.drive(OM_lin_speed, OM_ang_speed, OM_time)
    time.sleep(0.4)

    mirte.drive(LT_lin_speed, LT_scaler*LT_ang_speed, LT_time)

    time.sleep(0.4)

del mirte