import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../Mirte/ku_mirte_python'
    )
)

from ku_mirte import KU_Mirte

SIDE_SPEED = 0.35       # m/s
DISTANCE = 1.0          # meter

BASE_SPEED_MOD = 2.38
BASE_TURN_MOD = 2.38

# Tid = afstand / hastighed
SIDE_TIME = DISTANCE / SIDE_SPEED

mirte = KU_Mirte()
mirte.set_driving_modifier(
    BASE_SPEED_MOD,
    BASE_TURN_MOD
)


try:
    print(
        f"Kører sidelæns ca. {DISTANCE:.1f} meter "
        f"med hastighed {SIDE_SPEED:.2f} m/s."
    )

    # [x, y]
    # x = frem/bagud
    # y = sidelæns
    #
    # [0.0, +SIDE_SPEED] = sidelæns til venstre
    mirte.drive(
        [0.0, SIDE_SPEED],
        0.0,
        SIDE_TIME,
        blocking=True
    )

    mirte.stop()

    print("Færdig. Robotten er stoppet.")


except KeyboardInterrupt:
    print("\nAfbrudt.")


finally:
    mirte.stop()
    del mirte