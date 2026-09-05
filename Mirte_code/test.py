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
        OM_ang_speed = float(input('ang:'))
        OM_time = float(input('time:'))
        mirte.drive(OM_lin_speed, OM_ang_speed, OM_time)
del mirte