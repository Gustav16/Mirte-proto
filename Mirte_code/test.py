#getting path to import KU_Mirte
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../Mirte/ku_mirte_python'))

import time
from ku_mirte import KU_Mirte

#init mirte mirte
mirte = KU_Mirte()

#set driving modfier and start driving

mirte.drive(1:, 0, 1)


# ... jeres kode med mirte ...

del mirte