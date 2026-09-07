import sys
import os
import time

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../Mirte/ku_mirte_python'
    )
)

from Mirte.ku_mirte_python.ku_mirte import KU_Mirte


# -------------------------
# Settings
# -------------------------

SAFE_DISTANCE = 0.35
FWD_SPEED = 0.15
SPIN_RATE = 1.5


# -------------------------
# Start robot
# -------------------------

mirte = KU_Mirte()


try:
    while True:

        # Read sonar sensors
        sonar = mirte.sonar

        left = sonar["front_left"]
        right = sonar["front_right"]

        print(
            f"Left: {left:.2f} m | "
            f"Right: {right:.2f} m"
        )

        # No obstacle
        if left > SAFE_DISTANCE and right > SAFE_DISTANCE:

            mirte.drive(
                FWD_SPEED,
                0.0,
                None,
                blocking=False
            )

        # Obstacle -> choose the side with most space
        else:

            if left > right:

                # More room on the left
                mirte.drive(
                    0.0,
                    SPIN_RATE,
                    None,
                    blocking=False
                )

            else:

                # More room on the right
                mirte.drive(
                    0.0,
                    -SPIN_RATE,
                    None,
                    blocking=False
                )

        time.sleep(0.05)


except KeyboardInterrupt:
    print("Stopping robot")

finally:
    mirte.stop()
    del mirte