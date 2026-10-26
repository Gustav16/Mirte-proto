import math
import sys
from pathlib import Path

EX6_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EX6_DIR))

import camera
from ku_mirte import KU_Mirte


TARGET_ID = 6
LANDMARK_IDS = {1, 2, 3, 4, TARGET_ID}
SUCCESS_RANGE_CM = 30.0
STEP_DISTANCE_CM = 10.0
MAX_TOTAL_DISTANCE_CM = 1000.0
MAX_BEARING_DEGREES = 10.0
LINEAR_SPEED_MPS = 0.3

landmarks = {landmark_id: (0.0, 0.0) for landmark_id in LANDMARK_IDS}


def main():
    print("This test moves MIRTE forward in confirmed 5 cm steps.")
    print(f"Target ID: {TARGET_ID}; success range: {SUCCESS_RANGE_CM:.0f} cm.")
    print(f"Maximum total movement: {MAX_TOTAL_DISTANCE_CM:.0f} cm.")
    print("Run only with a clear path and a person ready to stop the robot.")
    if input('Type "START" to continue: ').strip() != "START":
        print("Test cancelled; no movement commanded.")
        return

    mirte = KU_Mirte()
    total_moved_cm = 0.0
    try:
        while total_moved_cm < MAX_TOTAL_DISTANCE_CM:
            z, local_landmarks, _ = camera.get_observations(landmarks, mirte)
            if TARGET_ID not in local_landmarks or TARGET_ID not in z:
                print(f"Target {TARGET_ID} is not visible; stopping without moving.")
                return

            distance_cm, bearing_rad = z[TARGET_ID]
            bearing_deg = math.degrees(float(bearing_rad))
            print(
                f"Target distance: {distance_cm:.1f} cm; "
                f"bearing: {bearing_deg:.1f} degrees."
            )

            if distance_cm <= SUCCESS_RANGE_CM:
                print("Target reached according to the camera distance threshold.")
                return

            if abs(bearing_deg) > MAX_BEARING_DEGREES:
                print("Target is not close enough to straight ahead; stopping.")
                return

            step_cm = min(
                STEP_DISTANCE_CM,
                MAX_TOTAL_DISTANCE_CM - total_moved_cm,
            )
            confirmation = input(
                f"Drive forward {step_cm:.0f} cm? Type 'y' to move, anything else to stop: "
            ).strip().lower()
            if confirmation != "y":
                print("Stopped by operator; no further movement commanded.")
                return

            mirte.drive(
                LINEAR_SPEED_MPS,
                0.0,
                (step_cm / 100.0) / LINEAR_SPEED_MPS,
                blocking=True,
            )
            total_moved_cm += step_cm

        print(
            f"Stopped at the {MAX_TOTAL_DISTANCE_CM:.0f} cm test movement limit. "
            "The target was not yet within the success range."
        )
    finally:
        mirte.stop()


if __name__ == "__main__":
    main()
