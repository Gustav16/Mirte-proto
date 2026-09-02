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
try:
    #ONE METER (2 cm more and little left turn)
    # mirte.drive(0.5, -0.003, 1.85) 

    scales = [0.8,0.85,0.9,0.95,1]
    i = 0 
    while(i < 5):
        mirte.drive(0, -1, scales[i]*math.pi/2)
        time.sleep(5)
        i += 1


    # ... jeres kode med mirte ...

except KeyboardInterrupt:
    print("Program interrupted!")

finally:
    mirte.close()