import math
import time

import numpy as np

from ku_mirte import KU_Mirte

from between import (
    find_safe_passage,
    goal_in_front_of_box,
    landmark_dict,
    limit_goal_distance,
)
from mirte_rrt_smooth import (
    make_map,
    plan_path,
    drive_path,
    save_path_plot,
)


# How close MIRTE should stop to the requested box.
STOP_DISTANCE = 0.40

# Exploration is deliberately done in short movements.
EXPLORE_DISTANCE = 0.40

# When no useful landmark is visible, turn and look again.
SEARCH_ANGLE = math.radians(30)
SEARCH_SPEED = 0.70

# Prevent endless exploration.
MAX_EXPLORE_STEPS = 20


def visible_ids(world_map):
    return [
        int(marker_id)
        for _, marker_id in world_map.landmarks
    ]


def goal_towards_one_box(world_map):
    """
    If only one useful box is visible, move a short distance towards it,
    but do not drive all the way to the obstacle.
    """
    points = landmark_dict(world_map.landmarks)

    if not points:
        return None, None

    # Prefer the closest visible box.
    marker_id = min(
        points,
        key=lambda i: np.linalg.norm(points[i])
    )

    box = points[marker_id]
    distance = np.linalg.norm(box)

    # Keep a little more clearance during exploration.
    safe_distance = (
        world_map.landmark_radius
        + world_map.mirte_radius
        + 0.15
    )

    if distance <= safe_distance:
        return None, marker_id

    goal = box * (
        (distance - safe_distance) / distance
    )

    goal = limit_goal_distance(
        goal,
        EXPLORE_DISTANCE
    )

    return goal, marker_id


def explore_once(mirte, world_map):
    """
    Choose one simple exploration action.

    2+ visible boxes:
        Try to move towards a safe passage between two boxes.

    1 visible box, or no safe passage:
        Move a short distance towards the closest box.

    0 visible boxes:
        Turn 30 degrees and look again.
    """
    count = len(world_map.landmarks)

    if count >= 2:
        passage_goal, pair = find_safe_passage(
            world_map
        )

        if passage_goal is not None:
            # Move towards the passage in short steps.
            goal = limit_goal_distance(
                passage_goal,
                EXPLORE_DISTANCE
            )

            print(
                f"Exploring towards passage between "
                f"ID {pair[0]} and ID {pair[1]}."
            )

            path = plan_path(
                world_map,
                goal
            )

            if path is not None:
                drive_path(
                    mirte,
                    world_map,
                    path
                )
                return True

    if count >= 1:
        goal, marker_id = goal_towards_one_box(
            world_map
        )

        if goal is not None:
            print(
                f"Exploring towards visible ID {marker_id}."
            )

            path = plan_path(
                world_map,
                goal
            )

            if path is not None:
                drive_path(
                    mirte,
                    world_map,
                    path
                )
                return True

        # A box is visible but there is no safe forward move.
        # Turn to search for another view.
        print("No safe exploration path. Turning 30 degrees.")

    else:
        print("No ArUco boxes visible. Turning 30 degrees.")

    scaler = 1.05
    mirte.drive(
        0.0,
        SEARCH_SPEED,
        scaler * SEARCH_ANGLE / SEARCH_SPEED,
    )

    time.sleep(0.3)
    return True


def main():
    try:
        target_id = int(
            input("ArUco ID to find and drive to: ")
        )
    except ValueError:
        print("Please enter a number.")
        return

    mirte = KU_Mirte()
    time.sleep(1)

    try:
        for step in range(MAX_EXPLORE_STEPS):
            print(
                f"\n--- Search step "
                f"{step + 1}/{MAX_EXPLORE_STEPS} ---"
            )

            # Every loop starts with a fresh local observation.
            world_map = make_map(mirte)
            ids = visible_ids(world_map)

            print("Visible ArUco IDs:", ids)

            # --------------------------------------------------
            # Target found
            # --------------------------------------------------
            if target_id in ids:
                print(f"Found target ID {target_id}.")

                goal = goal_in_front_of_box(
                    world_map,
                    target_id,
                    STOP_DISTANCE
                )

                if goal is None:
                    print("Could not calculate target goal.")
                    return

                if np.linalg.norm(goal) < 0.05:
                    print(
                        f"Already close enough to ID {target_id}."
                    )
                    return

                print(
                    f"Driving towards ID {target_id}. "
                    f"Stopping about {STOP_DISTANCE:.2f} m away."
                )

                path = plan_path(
                    world_map,
                    goal
                )

                if path is None:
                    print(
                        "Target is visible, but no safe path "
                        "was found. Continuing exploration."
                    )
                    explore_once(
                        mirte,
                        world_map
                    )
                    continue

                print("Path to target found.")

                drive_path(
                    mirte,
                    world_map,
                    path
                )

                save_path_plot(
                    world_map,
                    path,
                    "to_box_plan.png"
                )

                print(
                    f"Finished drive towards ID {target_id}."
                )
                return

            # --------------------------------------------------
            # Target not visible: move to get a new view
            # --------------------------------------------------
            print(
                f"Target ID {target_id} is not visible."
            )

            explore_once(
                mirte,
                world_map
            )

        print(
            f"Stopped after {MAX_EXPLORE_STEPS} "
            f"exploration steps without reaching ID {target_id}."
        )

    finally:
        try:
            mirte.drive(
                0.0,
                0.0,
                0.1
            )
        except Exception:
            pass

        del mirte


if __name__ == "__main__":
    main()
