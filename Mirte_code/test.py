#getting path to import KU_Mirte
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../Mirte/ku_mirte_python'))

import time
from ku_mirte import KU_Mirte

#init mirte mirte
mirte = KU_Mirte()

#set driving modfier and start driving
mirte.set_driving_modifier(speed_modifier=0.5, turn_modifier=0)
mirte.drive
start = time.perf_counter()

#non-blocking loop for driving 1 sec
while (mirte.is_driving):
    if (time.perf_counter() - start > 1):
        mirte.stop()




# ... jeres kode med mirte ...

del mirte