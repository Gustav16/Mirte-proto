import math
import time

import numpy as np
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../../Mirte/ku_mirte_python'
    )
)

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



def transform_local_path(path, pose):
    """
    Convert a local planned path to the simple global drawing frame.

    pose = [x, z, theta]
    theta = 0 means forward in +z.
    """
    x0, z0, theta = pose
    result = []

    for point in path:
        lx, lz = point

        gx = x0 + math.cos(theta) * lx - math.sin(theta) * lz
        gz = z0 + math.sin(theta) * lx + math.cos(theta) * lz

        result.append(np.array([gx, gz]))

    return result


def update_drawing_pose(pose, local_path):
    """
    Update the drawing pose from the end direction of a local planned path.
    This is only for the path drawing, not for robot control.
    """
    if len(local_path) < 2:
        return pose

    global_path = transform_local_path(local_path, pose)
    end = global_path[-1]

    a = local_path[-2]
    b = local_path[-1]
    dx = b[0] - a[0]
    dz = b[1] - a[1]

    local_heading = math.atan2(-dx, dz)

    return np.array([
        end[0],
        end[1],
        pose[2] + local_heading
    ])


def save_full_path(path_history, filename="full_search_path.png"):
    """Save the complete path that the planner intended to take."""
    if len(path_history) < 2:
        return

    xs = [p[0] for p in path_history]
    zs = [p[1] for p in path_history]

    plt.figure()
    plt.plot(xs, zs, "-r", linewidth=2, label="planned path")
    plt.plot(xs[0], zs[0], "xg", markersize=10, label="start")
    plt.plot(xs[-1], zs[-1], "xb", markersize=10, label="finish")

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Complete Planned Search Path")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved complete path to: {filename}")

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


def explore_once(
    mirte,
    world_map,
    path_history,
    drawing_pose,
):
    """
    Exploration rule:
    - 2+ boxes: plan and follow a complete route to a safe passage.
    - 0 or 1 box: rotate in place and scan again.
    """
    if len(world_map.landmarks) < 2:
        print(
            "Fewer than two usable boxes are visible. "
            "Turning 30 degrees and scanning again."
        )
        mirte.drive(
            0.0,
            SEARCH_SPEED,
            SEARCH_ANGLE / SEARCH_SPEED,
            blocking=True,
        )
        mirte.stop()
        drawing_pose[2] += SEARCH_ANGLE
        return True

    goal, pair = find_safe_passage(world_map)

    if goal is None:
        print(
            "No safe passage between the visible boxes. "
            "Turning 30 degrees and scanning again."
        )
        mirte.drive(
            0.0,
            SEARCH_SPEED,
            SEARCH_ANGLE / SEARCH_SPEED,
            blocking=True,
        )
        mirte.stop()
        drawing_pose[2] += SEARCH_ANGLE
        return True

    id_a, id_b = pair
    print(
        f"Planning complete route to passage "
        f"between ID {id_a} and ID {id_b}."
    )

    path = plan_path(world_map, goal)

    if path is None:
        print(
            "Could not find a safe route to the passage. "
            "Turning and scanning again."
        )
        mirte.drive(
            0.0,
            SEARCH_SPEED,
            SEARCH_ANGLE / SEARCH_SPEED,
            blocking=True,
        )
        mirte.stop()
        drawing_pose[2] += SEARCH_ANGLE
        return True

    global_segment = transform_local_path(path, drawing_pose)
    path_history.extend(global_segment[1:])
    update_drawing_pose(drawing_pose, path)

    print(
        "Following complete route to passage. "
        "Next scan happens after reaching it."
    )

    reached = drive_path(mirte, world_map, path)

    if reached is False:
        print(
            "Route stopped early by safety system. "
            "Taking a new scan."
        )

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

    # Used only for drawing the complete planned route.
    path_history = [np.array([0.0, 0.0])]
    drawing_pose = np.array([0.0, 0.0, 0.0])

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
                        world_map,
                        path_history,
                        drawing_pose
                    )
                    continue

                print("Path to target found.")

                global_segment = transform_local_path(
                    path,
                    drawing_pose
                )
                path_history.extend(global_segment[1:])

                # Local map + final local RRT path.
                save_path_plot(
                    world_map,
                    path,
                    "to_box_plan.png"
                )

                # Whole planned route from the original start.
                save_full_path(
                    path_history,
                    "full_search_path.png"
                )

                drive_path(
                    mirte,
                    world_map,
                    path
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
                world_map,
                path_history,
                drawing_pose
            )

        print(
            f"Stopped after {MAX_EXPLORE_STEPS} "
            f"exploration steps without reaching ID {target_id}."
        )

        save_full_path(
            path_history,
            "full_search_path.png"
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
