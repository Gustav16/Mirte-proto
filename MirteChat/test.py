from Mirte.ku_mirte_python.ku_mirte import KU_Mirte
import time
mirte = KU_Mirte()

mirte.set_driving_modifier(speed_modifier=0.5, turn_modifier=0)
mirte.drive
start = time.perf_counter()


while (mirte.is_driving):
    if (time.perf_counter() - start > 1):
        mirte.stop
        #stop
        break




# ... jeres kode med mirte ...

del mirte