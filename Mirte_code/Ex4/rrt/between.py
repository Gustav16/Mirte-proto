import math
import os
import sys
import time

import numpy as np

# Use a non-interactive backend on the robot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# MIRTE imports
# ---------------------------------------------------------------------------

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../Mirte/ku_mirte_python"
    )
)

from ku_mirte import KU_Mirte
from local_map import LocalMap


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# If you know the two exact IDs, set e.g.:
# TARGET_IDS = (1, 10)
#
# None = choose the two visible markers with the largest horizontal separation.
TARGET_IDS = None

LINEAR_SPEED = 0.15       # m/s
ANGULAR_SPEED = 0.35      # rad/s

# Calibration multipliers.
# Keep at 1.0 initially.
# If the robot systematically drives too short/far, adjust DISTANCE_SCALE.
# If it systematically turns too little/much, adjust TURN_SCALE.
DISTANCE_SCALE = 1.0
TURN_SCALE = 1.0

# 0.0 means robot center aims directly for the calculated midpoint.
# Set e.g. 0.10 to stop 10 cm before it.
STOP_BEFORE_MIDPOINT = 0.0

# Simple safety limit.
MAX_DRIVE_DISTANCE = 3.0

PLOT_FILE = "between_plan.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stop_robot(mirte):
    try:
        mirte.drive(0.0, 0.0, 0.1)
    except Exception:
        pass


def landmark_dict(landmarks):
    """
    LocalMap gives:
        [[np.array([x, z]), id], ...]

    Convert to:
        {id: np.array([x, z]), ...}

    Coordinates:
        x < 0 : left of robot
        x > 0 : right of robot
        z > 0 : in front of robot
    """
    points = {}

    for position, marker_id in landmarks:
        points[int(marker_id)] = np.asarray(position, dtype=float)

    return points


def print_landmarks(points):
    print("Detected ArUco landmarks:")

    for marker_id in sorted(points):
        x, z = points[marker_id]
        print(
            f"  ID {marker_id:3d}: "
            f"x = {x:+.3f} m, z = {z:+.3f} m"
        )


def choose_marker_pair(points):
    """
    Use TARGET_IDS if specified.

    Otherwise choose the two markers with the largest separation
    in the camera/robot x direction. This works well when the two
    landmarks form the left and right side of a passage.
    """
    if TARGET_IDS is not None:
        id_a, id_b = map(int, TARGET_IDS)

        if id_a not in points or id_b not in points:
            raise RuntimeError(
                f"Could not see both target IDs {id_a} and {id_b}. "
                "Reposition MIRTE and run the program again."
            )

        return id_a, id_b

    ids = list(points.keys())

    if len(ids) < 2:
        raise RuntimeError(
            "Fewer than two ArUco markers were detected. "
            "Reposition MIRTE so both markers are visible and run again."
        )

    best_pair = None
    best_separation = -1.0

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a = ids[i]
            id_b = ids[j]

            separation = abs(points[id_a][0] - points[id_b][0])

            if separation > best_separation:
                best_separation = separation
                best_pair = (id_a, id_b)

    return best_pair


def calculate_plan(points, marker_pair):
    id_a, id_b = marker_pair

    p_a = points[id_a]
    p_b = points[id_b]

    midpoint = (p_a + p_b) / 2.0

    x = float(midpoint[0])
    z = float(midpoint[1])

    # Euclidean distance from robot origin (0, 0) to midpoint.
    distance = math.hypot(x, z)

    # Camera/LocalMap x is positive to the right.
    # Positive robot yaw is left/counter-clockwise, hence the minus sign.
    turn_angle = -math.atan2(x, z)

    gap_width = float(np.linalg.norm(p_a - p_b))

    return midpoint, distance, turn_angle, gap_width


def save_plan_plot(points, marker_pair, midpoint, filename=PLOT_FILE):
    """
    Save a plot BEFORE moving.

    Robot start = (0, 0)
    Landmarks = their measured [x, z] positions
    Goal = midpoint between the selected landmarks
    """
    id_a, id_b = marker_pair

    fig, ax = plt.subplots(figsize=(7, 8))

    # Plot all detected landmarks.
    for marker_id, point in points.items():
        x, z = point

        if marker_id in marker_pair:
            marker = "s"
            size = 100
        else:
            marker = "o"
            size = 60

        ax.scatter(x, z, marker=marker, s=size)
        ax.text(
            x + 0.025,
            z + 0.025,
            f"ID {marker_id}",
            fontsize=10
        )

    # Robot start.
    ax.scatter(0.0, 0.0, marker="^", s=120)
    ax.text(0.025, 0.025, "MIRTE start", fontsize=10)

    # Midpoint / target.
    ax.scatter(midpoint[0], midpoint[1], marker="x", s=140)
    ax.text(
        midpoint[0] + 0.025,
        midpoint[1] + 0.025,
        "Target midpoint",
        fontsize=10
    )

    # Line between selected landmarks.
    p_a = points[id_a]
    p_b = points[id_b]
    ax.plot(
        [p_a[0], p_b[0]],
        [p_a[1], p_b[1]],
        linestyle="--"
    )

    # Planned robot path.
    ax.plot(
        [0.0, midpoint[0]],
        [0.0, midpoint[1]],
        linestyle="-"
    )

    ax.set_xlabel("x [m]  (left / right)")
    ax.set_ylabel("z [m]  (forward)")
    ax.set_title(
        f"One-shot plan between ArUco ID {id_a} and ID {id_b}"
    )
    ax.grid(True)
    ax.axis("equal")

    # Make sure origin is visible with some margin.
    all_x = [0.0, midpoint[0]] + [p[0] for p in points.values()]
    all_z = [0.0, midpoint[1]] + [p[1] for p in points.values()]

    x_margin = 0.25
    z_margin = 0.25

    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_z) - z_margin, max(all_z) + z_margin)

    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def rotate_robot(mirte, angle):
    corrected_angle = angle * TURN_SCALE

    if abs(corrected_angle) < math.radians(0.5):
        print("Rotation is negligible. Skipping turn.")
        return

    angular_velocity = math.copysign(
        ANGULAR_SPEED,
        corrected_angle
    )

    duration = abs(corrected_angle) / ANGULAR_SPEED

    print(
        f"Rotating {math.degrees(corrected_angle):+.2f} degrees "
        f"for {duration:.2f} s"
    )

    mirte.drive(
        0.0,
        angular_velocity,
        duration
    )


def drive_to_target(mirte, distance):
    drive_distance = max(
        0.0,
        distance - STOP_BEFORE_MIDPOINT
    )

    drive_distance *= DISTANCE_SCALE

    if drive_distance <= 0.0:
        print("No forward movement is needed.")
        return

    if drive_distance > MAX_DRIVE_DISTANCE:
        raise RuntimeError(
            f"Calculated drive distance is {drive_distance:.2f} m, "
            f"which exceeds safety limit {MAX_DRIVE_DISTANCE:.2f} m."
        )

    duration = drive_distance / LINEAR_SPEED

    print(
        f"Driving {drive_distance:.3f} m forward "
        f"for {duration:.2f} s"
    )

    mirte.drive(
        LINEAR_SPEED,
        0.0,
        duration
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mirte = None

    try:
        print("Starting MIRTE...")
        mirte = KU_Mirte()

        # Let camera / ROS initialize.
        time.sleep(1.0)

        print()
        print("Taking ONE ArUco map...")

        local_map = LocalMap()

        # IMPORTANT:
        # This is the only camera/map update in the program.
        local_map.update(mirte)

        points = landmark_dict(local_map.landmarks)

        if not points:
            raise RuntimeError(
                "No ArUco markers were detected. "
                "Reposition MIRTE and run the program again."
            )

        print_landmarks(points)

        marker_pair = choose_marker_pair(points)
        id_a, id_b = marker_pair

        print()
        print(f"Selected target markers: ID {id_a} and ID {id_b}")

        midpoint, distance, turn_angle, gap_width = calculate_plan(
            points,
            marker_pair
        )

        print()
        print("Calculated one-shot plan:")
        print(f"  Distance between markers : {gap_width:.3f} m")
        print(
            f"  Midpoint                 : "
            f"x={midpoint[0]:+.3f} m, z={midpoint[1]:+.3f} m"
        )
        print(f"  Distance to midpoint     : {distance:.3f} m")
        print(
            f"  Required turn            : "
            f"{math.degrees(turn_angle):+.2f} degrees"
        )

        if midpoint[1] <= 0.0:
            raise RuntimeError(
                "The calculated midpoint is not in front of MIRTE. "
                "For this test, place MIRTE in front of the two markers."
            )

        save_plan_plot(
            points,
            marker_pair,
            midpoint
        )

        print()
        print(f"Saved plan plot to: {os.path.abspath(PLOT_FILE)}")

        print()
        print("Camera measurements are now finished.")
        print("MIRTE will execute this fixed estimate without looking again.")

        # Step 1: rotate once to face the originally calculated midpoint.
        rotate_robot(
            mirte,
            turn_angle
        )

        time.sleep(0.25)

        # Step 2: drive the originally calculated distance.
        drive_to_target(
            mirte,
            distance
        )

        # Explicit stop.
        stop_robot(mirte)

        print()
        print("Finished.")
        print("MIRTE has stopped after executing the original one-shot plan.")

    except KeyboardInterrupt:
        print()
        print("Ctrl+C received. Stopping MIRTE.")

    except Exception as exc:
        print()
        print("ERROR:")
        print(exc)
        raise

    finally:
        if mirte is not None:
            stop_robot(mirte)

        print("Robot stopped.")


if __name__ == "__main__":
    main()
