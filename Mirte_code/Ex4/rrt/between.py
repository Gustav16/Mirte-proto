"""
between_v2.py

One camera map -> choose a passage -> plan -> execute -> stop.

Changes from the first between.py:
1. Uses local_map.py, where ArUco positions are relative to MIRTE's center.
2. Slightly higher translation speed: 0.18 m/s instead of 0.15 m/s.
3. Uses MIRTE's holonomic/mecanum movement:
       mirte.drive([forward_speed, sideways_speed], ...)
   so MIRTE can move diagonally/sideways instead of relying on a tiny
   timed rotation.
4. Detects ALL visible ArUco boxes.
5. With 3+ boxes, all boxes are treated as obstacles.
6. If the direct path is blocked, it uses the existing RRT code to find
   a collision-free route.
7. The camera is used only ONCE. There is no re-detection while driving.

Files expected in the same folder:
    between_v2.py
    local_map.py
    rrt.py
    robot_models.py
"""

import math
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../Mirte/ku_mirte_python"
    )
)

from ku_mirte import KU_Mirte

from local_map import LocalMap
from robot_models import PointMassModel
from rrt import RRT


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# If you know which two markers form the gate that MIRTE should pass through:
#
#     TARGET_IDS = (1, 10)
#
# If None:
# - exactly 2 markers -> those two are used
# - 3+ markers -> an automatic central/passable gap is selected
TARGET_IDS = None


# Slight speed increase from 0.15 m/s.
LINEAR_SPEED = 0.18

# Stop exactly at the calculated midpoint.
# Set e.g. 0.05 to stop 5 cm before it.
STOP_BEFORE_GOAL = 0.00


# RRT settings.
PATH_RESOLUTION = 0.05
EXPAND_DISTANCE = 0.25
MAX_RRT_ITER = 2000
GOAL_SAMPLE_RATE = 10

# Extra clearance beyond LocalMap's robot + landmark radii.
GATE_EXTRA_MARGIN = 0.05

# Plot output.
PLOT_FILE = "between_plan_v2.png"


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def stop_robot(mirte):
    try:
        mirte.stop()
    except Exception:
        try:
            mirte.drive([0.0, 0.0], 0.0, 0.1)
        except Exception:
            pass


def landmark_dict(landmarks):
    points = {}

    for position, marker_id in landmarks:
        points[int(marker_id)] = np.asarray(
            position,
            dtype=float
        )

    return points


def print_landmarks(points):
    print("Detected ArUco landmarks in ROBOT-CENTER frame:")

    for marker_id in sorted(points):
        x, z = points[marker_id]

        print(
            f"  ID {marker_id:3d}: "
            f"x = {x:+.3f} m, "
            f"z = {z:+.3f} m"
        )


def midpoint(a, b):
    return (
        np.asarray(a, dtype=float)
        + np.asarray(b, dtype=float)
    ) / 2.0


def bearing_to(point):
    """
    Bearing from forward direction.

    Map:
        x positive = right
        z positive = forward
    """
    x, z = point

    return math.atan2(
        -float(x),
        float(z)
    )


# ---------------------------------------------------------------------------
# Gate selection
# ---------------------------------------------------------------------------

def gate_is_wide_enough(
    local_map,
    p_a,
    p_b
):
    center_distance = float(
        np.linalg.norm(p_a - p_b)
    )

    required_center_distance = (
        2.0
        * (
            local_map.landmark_radius
            + local_map.mirte_radius
        )
        + GATE_EXTRA_MARGIN
    )

    return (
        center_distance
        >= required_center_distance
    )


def automatic_gate_candidates(points):
    """
    For 3+ markers, consider only neighbours in viewing angle.

    This avoids choosing e.g. the far-left and far-right box as one
    giant "gate" while another box is actually between them.
    """
    ids = list(points.keys())

    sorted_ids = sorted(
        ids,
        key=lambda marker_id: bearing_to(
            points[marker_id]
        )
    )

    return [
        (
            sorted_ids[i],
            sorted_ids[i + 1]
        )
        for i in range(
            len(sorted_ids) - 1
        )
    ]


def choose_gate(
    points,
    local_map
):
    ids = list(points.keys())

    if len(ids) < 2:
        raise RuntimeError(
            "At least two ArUco landmarks are required."
        )

    # Explicit gate.
    if TARGET_IDS is not None:
        id_a, id_b = map(
            int,
            TARGET_IDS
        )

        if (
            id_a not in points
            or id_b not in points
        ):
            raise RuntimeError(
                f"Could not see both requested target IDs "
                f"{id_a} and {id_b}."
            )

        p_a = points[id_a]
        p_b = points[id_b]
        goal = midpoint(p_a, p_b)

        if not gate_is_wide_enough(
            local_map,
            p_a,
            p_b
        ):
            raise RuntimeError(
                f"Gate between ID {id_a} and ID {id_b} "
                "is too narrow according to the current "
                "robot/landmark radii."
            )

        if local_map.in_collision(goal):
            raise RuntimeError(
                "The midpoint of the requested gate is "
                "inside the collision region."
            )

        return (
            id_a,
            id_b,
            goal
        )

    # Exactly two -> obvious gate.
    if len(ids) == 2:
        id_a, id_b = ids

        p_a = points[id_a]
        p_b = points[id_b]
        goal = midpoint(p_a, p_b)

        if not gate_is_wide_enough(
            local_map,
            p_a,
            p_b
        ):
            raise RuntimeError(
                "The two detected landmarks do not leave "
                "enough clearance for MIRTE."
            )

        return (
            id_a,
            id_b,
            goal
        )

    # 3+ landmarks -> only adjacent visual neighbours are candidate gates.
    candidates = []

    for id_a, id_b in automatic_gate_candidates(
        points
    ):
        p_a = points[id_a]
        p_b = points[id_b]

        if not gate_is_wide_enough(
            local_map,
            p_a,
            p_b
        ):
            continue

        goal = midpoint(
            p_a,
            p_b
        )

        # Must be in front of MIRTE.
        if goal[1] <= 0.0:
            continue

        # Goal itself must be safe.
        if local_map.in_collision(goal):
            continue

        distance = float(
            np.linalg.norm(goal)
        )

        angle = abs(
            bearing_to(goal)
        )

        # Prefer a gate near the forward direction.
        # Distance is a smaller secondary preference.
        score = (
            3.0 * angle
            + 0.15 * distance
        )

        candidates.append(
            (
                score,
                id_a,
                id_b,
                goal
            )
        )

    if not candidates:
        raise RuntimeError(
            "No automatically passable gap was found. "
            "Set TARGET_IDS explicitly or change the layout."
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    _, id_a, id_b, goal = candidates[0]

    return (
        id_a,
        id_b,
        goal
    )


# ---------------------------------------------------------------------------
# Collision checking / RRT
# ---------------------------------------------------------------------------

def segment_is_free(
    local_map,
    p0,
    p1,
    step=0.025
):
    p0 = np.asarray(
        p0,
        dtype=float
    )

    p1 = np.asarray(
        p1,
        dtype=float
    )

    distance = float(
        np.linalg.norm(
            p1 - p0
        )
    )

    if distance <= 1e-9:
        return not bool(
            local_map.in_collision(p0)
        )

    n = max(
        2,
        int(
            math.ceil(
                distance / step
            )
        )
        + 1
    )

    for t in np.linspace(
        0.0,
        1.0,
        n
    ):
        p = (
            (1.0 - t) * p0
            + t * p1
        )

        if local_map.in_collision(p):
            return False

    return True


def simplify_path(
    path,
    local_map
):
    """
    Greedy line-of-sight path simplification.

    This reduces the number of physical MIRTE movement commands,
    which should reduce accumulated open-loop motion error.
    """
    if len(path) <= 2:
        return path

    simplified = [
        np.asarray(
            path[0],
            dtype=float
        )
    ]

    i = 0

    while i < len(path) - 1:
        chosen = i + 1

        for j in range(
            len(path) - 1,
            i,
            -1
        ):
            if segment_is_free(
                local_map,
                path[i],
                path[j]
            ):
                chosen = j
                break

        simplified.append(
            np.asarray(
                path[chosen],
                dtype=float
            )
        )

        i = chosen

    return simplified


def create_path(
    local_map,
    goal
):
    start = np.array([
        0.0,
        0.0
    ])

    goal = np.asarray(
        goal,
        dtype=float
    )

    # First try the simple direct route.
    if segment_is_free(
        local_map,
        start,
        goal
    ):
        print(
            "Direct path to selected midpoint is collision-free."
        )

        return [
            start,
            goal
        ]

    print(
        "Direct route is blocked by another box."
    )

    print(
        "Running RRT using all detected boxes as obstacles..."
    )

    robot_model = PointMassModel(
        ctrl_range=[
            -PATH_RESOLUTION,
            PATH_RESOLUTION
        ]
    )

    planner = RRT(
        start=start,
        goal=goal,
        robot_model=robot_model,
        map=local_map,
        expand_dis=EXPAND_DISTANCE,
        path_resolution=PATH_RESOLUTION,
        goal_sample_rate=GOAL_SAMPLE_RATE,
        max_iter=MAX_RRT_ITER
    )

    raw_path = planner.planning(
        animation=False
    )

    if raw_path is None:
        raise RuntimeError(
            "RRT could not find a collision-free route."
        )

    # rrt.py returns goal -> ... -> start.
    path = [
        np.asarray(
            p,
            dtype=float
        )
        for p in reversed(raw_path)
    ]

    path = simplify_path(
        path,
        local_map
    )

    print(
        f"RRT path simplified to "
        f"{len(path)} waypoints."
    )

    return path


# ---------------------------------------------------------------------------
# Physical movement
# ---------------------------------------------------------------------------

def execute_segment(
    mirte,
    p0,
    p1
):
    """
    Execute a map-frame translation WITHOUT rotating MIRTE.

    Map coordinates:
        dx > 0 : target lies to robot's right
        dz > 0 : target lies forward

    KU_Mirte drive([x, y], ...):
        x > 0 : forward
        y > 0 : left

    Therefore:
        forward command = +dz
        sideways command = -dx
    """
    p0 = np.asarray(
        p0,
        dtype=float
    )

    p1 = np.asarray(
        p1,
        dtype=float
    )

    dx = float(
        p1[0] - p0[0]
    )

    dz = float(
        p1[1] - p0[1]
    )

    distance = math.hypot(
        dx,
        dz
    )

    if distance <= 1e-6:
        return

    usable_distance = distance

    # Only shorten the final segment.
    # Caller handles whether this is final by modifying p1 if needed.
    direction_forward = dz / distance
    direction_left = -dx / distance

    v_forward = (
        LINEAR_SPEED
        * direction_forward
    )

    v_left = (
        LINEAR_SPEED
        * direction_left
    )

    duration = (
        usable_distance
        / LINEAR_SPEED
    )

    print(
        "Moving segment:"
    )

    print(
        f"  dx(right) = {dx:+.3f} m"
    )

    print(
        f"  dz(forward) = {dz:+.3f} m"
    )

    print(
        f"  forward speed = {v_forward:+.3f} m/s"
    )

    print(
        f"  sideways(left) speed = {v_left:+.3f} m/s"
    )

    print(
        f"  duration = {duration:.2f} s"
    )

    mirte.drive(
        [
            v_forward,
            v_left
        ],
        0.0,
        duration
    )


def shorten_final_goal(
    path,
    stop_before
):
    if (
        stop_before <= 0.0
        or len(path) < 2
    ):
        return path

    new_path = [
        np.asarray(
            p,
            dtype=float
        ).copy()
        for p in path
    ]

    p0 = new_path[-2]
    p1 = new_path[-1]

    delta = p1 - p0

    distance = float(
        np.linalg.norm(delta)
    )

    if distance <= stop_before:
        return new_path[:-1]

    new_path[-1] = (
        p1
        - (
            delta / distance
        )
        * stop_before
    )

    return new_path


def execute_path(
    mirte,
    path
):
    for i in range(
        len(path) - 1
    ):
        print()
        print(
            f"Executing path segment "
            f"{i + 1}/{len(path) - 1}"
        )

        execute_segment(
            mirte,
            path[i],
            path[i + 1]
        )

        time.sleep(0.20)

    stop_robot(mirte)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plan_plot(
    local_map,
    points,
    gate_ids,
    goal,
    path
):
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    clearance = (
        local_map.landmark_radius
        + local_map.mirte_radius
    )

    for marker_id, point in points.items():
        x, z = point

        ax.scatter(
            x,
            z,
            marker="s"
        )

        ax.text(
            x + 0.025,
            z + 0.025,
            f"ID {marker_id}"
        )

        ax.add_patch(
            Circle(
                (x, z),
                clearance,
                fill=False
            )
        )

    # MIRTE start.
    ax.scatter(
        0.0,
        0.0,
        marker="^"
    )

    ax.text(
        0.025,
        0.025,
        "MIRTE start"
    )

    # Selected gate.
    id_a, id_b = gate_ids

    p_a = points[id_a]
    p_b = points[id_b]

    ax.plot(
        [p_a[0], p_b[0]],
        [p_a[1], p_b[1]],
        linestyle="--"
    )

    # Goal.
    ax.scatter(
        goal[0],
        goal[1],
        marker="x",
        s=100
    )

    ax.text(
        goal[0] + 0.025,
        goal[1] + 0.025,
        "Selected midpoint"
    )

    # Planned path.
    xs = [
        p[0]
        for p in path
    ]

    zs = [
        p[1]
        for p in path
    ]

    ax.plot(
        xs,
        zs,
        linewidth=2
    )

    ax.set_xlabel(
        "x [m] (right positive)"
    )

    ax.set_ylabel(
        "z [m] (forward positive)"
    )

    ax.set_title(
        f"Plan through ArUco ID {id_a} and ID {id_b}"
    )

    ax.grid(True)

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    all_x = (
        [0.0, goal[0]]
        + [
            p[0]
            for p in points.values()
        ]
        + xs
    )

    all_z = (
        [0.0, goal[1]]
        + [
            p[1]
            for p in points.values()
        ]
        + zs
    )

    margin = 0.60

    ax.set_xlim(
        min(all_x) - margin,
        max(all_x) + margin
    )

    ax.set_ylim(
        min(0.0, min(all_z)) - 0.10,
        max(all_z) + margin
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_FILE,
        dpi=180
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mirte = None

    try:
        print(
            "Starting MIRTE..."
        )

        mirte = KU_Mirte()

        time.sleep(1.0)

        print()
        print(
            "Taking ONE ArUco map..."
        )

        local_map = LocalMap()

        # The only camera observation.
        local_map.update(mirte)

        points = landmark_dict(
            local_map.landmarks
        )

        if len(points) < 2:
            raise RuntimeError(
                "Fewer than two ArUco landmarks detected."
            )

        print_landmarks(points)

        id_a, id_b, goal = choose_gate(
            points,
            local_map
        )

        p_a = points[id_a]
        p_b = points[id_b]

        gap_width = float(
            np.linalg.norm(
                p_a - p_b
            )
        )

        goal_distance = float(
            np.linalg.norm(goal)
        )

        goal_angle = math.degrees(
            bearing_to(goal)
        )

        print()
        print(
            f"Selected gate: "
            f"ID {id_a} <-> ID {id_b}"
        )

        print(
            f"Center distance between boxes: "
            f"{gap_width:.3f} m"
        )

        print(
            f"Goal midpoint in ROBOT-CENTER frame: "
            f"x={goal[0]:+.3f} m, "
            f"z={goal[1]:+.3f} m"
        )

        print(
            f"Straight-line distance from robot center: "
            f"{goal_distance:.3f} m"
        )

        print(
            f"Goal bearing: "
            f"{goal_angle:+.2f} degrees "
            f"(positive = right)"
        )

        path = create_path(
            local_map,
            goal
        )

        path = shorten_final_goal(
            path,
            STOP_BEFORE_GOAL
        )

        save_plan_plot(
            local_map,
            points,
            (id_a, id_b),
            goal,
            path
        )

        print()
        print(
            f"Saved plan plot to: "
            f"{os.path.abspath(PLOT_FILE)}"
        )

        print()
        print(
            "Camera measurements are now finished."
        )

        print(
            "MIRTE will execute the fixed plan "
            "without looking again."
        )

        execute_path(
            mirte,
            path
        )

        print()
        print(
            "Finished. MIRTE stopped."
        )

    except KeyboardInterrupt:
        print()
        print(
            "Ctrl+C received. Stopping MIRTE."
        )

    except Exception as exc:
        print()
        print(
            "ERROR:"
        )

        print(exc)

        raise

    finally:
        if mirte is not None:
            stop_robot(mirte)

        print(
            "Robot stopped."
        )


if __name__ == "__main__":
    main()
