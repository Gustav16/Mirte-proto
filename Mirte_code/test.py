#getting path to import KU_Mirte
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../Mirte/ku_mirte_python'))

import time
from ku_mirte import KU_Mirte

#init mirte mirte
mirte = KU_Mirte()

#set driving modfier and start driving
try:
    mirte.drive(0.5, 0, 2)

    # ... jeres kode med mirte ...

except KeyboardInterrupt:
    print("Program interrupted!")

finally:
    mirte.close()