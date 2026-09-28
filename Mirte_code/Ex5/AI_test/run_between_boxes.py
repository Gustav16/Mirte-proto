import time

import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../../Mirte/ku_mirte_python'
    )
)

from ku_mirte import KU_Mirte



from between import goal_between_ids
from mirte_rrt_smooth import make_map, plan_path, drive_path, save_path_plot


# Change these two numbers if another pair of boxes should be used.
BOX_ID_1 = 1
BOX_ID_2 = 10


def main():
    mirte = KU_Mirte()
    time.sleep(1)

    try:
        world_map = make_map(mirte)

        visible_ids = [
            int(marker_id)
            for _, marker_id in world_map.landmarks
        ]

        print("Visible ArUco IDs:", visible_ids)

        goal = goal_between_ids(
            world_map,
            BOX_ID_1,
            BOX_ID_2
        )

        if goal is None:
            print(
                f"Could not see both ID {BOX_ID_1} "
                f"and ID {BOX_ID_2}."
            )
            return

        print(
            f"Driving between ID {BOX_ID_1} "
            f"and ID {BOX_ID_2}."
        )
        print(
            f"Goal: x={goal[0]:+.3f}, "
            f"z={goal[1]:+.3f}"
        )

        path = plan_path(world_map, goal)

        if path is None:
            print("Could not find a path.")
            return

        print("Path found.")
        drive_path(mirte, world_map, path)
        save_path_plot(
            world_map,
            path,
            "between_boxes_plan.png"
        )

    finally:
        # Stop command before cleanup.
        try:
            mirte.drive(0.0, 0.0, 0.1)
        except Exception:
            pass
        del mirte


if __name__ == "__main__":
    main()
