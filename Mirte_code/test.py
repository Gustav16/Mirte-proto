#getting path to import KU_Mirte
import sys
import os
import math
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../Mirte/ku_mirte_python'))

import time
from ku_mirte import KU_Mirte

#init mirte mirte
mirte = KU_Mirte()

#set driving modfier and start driving

while (True):
    msg = input('press (q) to quit, or press any other key to drive in a square:\n')
    if msg == 'q':
        break
    else:
        OM_lin_speed = float(input('speed:'))
        OM_ang_speed = float(input('ang:'))
        OM_time = float('time:')
        mirte.drive(OM_lin_speed, OM_ang_speed, OM_time)