"""No-motion geometry check.

Run this before enabling Exercise-5 driving. It prints both the ArUco marker
centre and the inferred physical box centre. The box centre uses the existing
Exercise-4 negative local-Z offset.
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../Mirte/ku_mirte_python"))
from robot_io import KU_Mirte

from aruco_measurements import observe_geometry_mirte
from ex5_config import LANDMARKS


def main():
    mirte = KU_Mirte()
    time.sleep(1.0)
    try:
        geometries = observe_geometry_mirte(mirte, allowed_ids=LANDMARKS.keys())
        if not geometries:
            print("No configured ArUco landmarks detected.")
            return

        for g in geometries:
            mx, mz = g["marker_position"]
            bx, bz = g["box_center"]
            print(
                f"ID {g['id']}: "
                f"ArUco(robot frame)=({mx:+.3f}, {mz:+.3f}) m | "
                f"box center=({bx:+.3f}, {bz:+.3f}) m | "
                f"delta=({bx-mx:+.3f}, {bz-mz:+.3f}) m"
            )
    finally:
        del mirte


if __name__ == "__main__":
    main()
