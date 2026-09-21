#Mirte Proto code

import time
import sys
import os
import json
import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --------------------------------------------------
# Import KU_Mirte and LocalMap
# --------------------------------------------------

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../../Mirte/ku_mirte_python'
    )
)

from ku_mirte import KU_Mirte
from local_map import LocalMap


# --------------------------------------------------
# Output files
# --------------------------------------------------

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(OUTPUT_DIR, "local_map.json")
PNG_PATH = os.path.join(OUTPUT_DIR, "local_map.png")

MIRTE_RADIUS = 0.20   # 20 cm
LANDMARK_RADIUS = 0.30  # 30 cm

# 2x2 m local map, centered on mirte.
MAP_XLIM = (-1, 1)
MAP_YLIM = (0, 2)
GRID_STEP = 0.25  # 25 cm


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def plot_local_map(landmarks):
    plt.clf()
    ax = plt.gca()

    # Mirte position
    ax.add_patch(Circle((0, 0), MIRTE_RADIUS, color='blue', alpha=0.3))
    plt.scatter(0, 0)
    plt.text(0, 0, "Mirte", fontsize=10)

    # Landmarks
    for position, landmark_id in landmarks:
        x, z = position

        ax.add_patch(Circle((x, z), LANDMARK_RADIUS, color='red', alpha=0.2))
        plt.scatter(x, z)
        plt.text(
            x,
            z,
            f"ID {landmark_id}",
            fontsize=10
        )

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Local Landmark Map")

    # Fixed local map area, gridlines every GRID_STEP meters
    plt.xlim(*MAP_XLIM)
    plt.ylim(*MAP_YLIM)
    ax.set_xticks(np.arange(MAP_XLIM[0], MAP_XLIM[1] + 1e-9, GRID_STEP))
    ax.set_yticks(np.arange(MAP_YLIM[0], MAP_YLIM[1] + 1e-9, GRID_STEP))
    plt.grid(True)

    # Equal geometric scale
    ax.set_aspect("equal", adjustable="box")

    plt.pause(0.01)


def plot_path(path, start=None, goal=None):
    """
    Draw an RRT route on top of the currently plotted local map.

    path: list of [x, z] points, e.g. RRT.generate_final_course()'s result.
    """
    ax = plt.gca()

    xs = [p[0] for p in path]
    zs = [p[1] for p in path]
    ax.plot(xs, zs, '-r', linewidth=2, label='path')

    if start is not None:
        ax.plot(start[0], start[1], 'xg', markersize=10, label='start')
    if goal is not None:
        ax.plot(goal[0], goal[1], 'xb', markersize=10, label='goal')

    ax.legend(loc='upper right')
    plt.pause(0.01)


# --------------------------------------------------
# Save map
# --------------------------------------------------

def save_map_json(landmarks, filename):
    data = []

    for position, landmark_id in landmarks:
        x, z = position

        data.append({
            "id": int(landmark_id),
            "x_m": float(x),
            "z_m": float(z)
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved map data to: {filename}")


if __name__ == "__main__":

    # ----------------------------------------------
    # Start MIRTE
    # ----------------------------------------------

    mirte = KU_Mirte()
    time.sleep(1)

    local_map = LocalMap()

    # ----------------------------------------------
    # Ctrl+C handling
    # ----------------------------------------------

    running = True

    def handle_sigint(signum, frame):
        global running

        print(
            "\nCtrl+C received. "
            "Finishing current iteration and saving map..."
        )

        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    # ----------------------------------------------
    # Interactive plot
    # ----------------------------------------------

    plt.ion()

    # ----------------------------------------------
    # Main loop
    # ----------------------------------------------

    try:

        while running:

            local_map.update(mirte)

            print(local_map.landmarks)

            plot_local_map(local_map.landmarks)

            time.sleep(0.3)

    finally:

        print("\nSaving final map...")

        save_map_json(
            local_map.landmarks,
            JSON_PATH
        )

        plt.savefig(
            PNG_PATH,
            dpi=200,
            bbox_inches="tight"
        )

        print(f"Saved plot to: {PNG_PATH}")

        plt.close("all")

        del mirte

        print("Finished.")
