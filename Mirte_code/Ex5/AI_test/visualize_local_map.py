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

from robot_io import KU_Mirte
from local_map import LocalMap


# --------------------------------------------------
# Output files
# --------------------------------------------------

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(OUTPUT_DIR, "local_map.json")
PNG_PATH = os.path.join(OUTPUT_DIR, "local_map.png")

MIRTE_RADIUS = 0.22   # 22 cm, keep in sync with LocalMap.mirte_radius
LANDMARK_RADIUS = 0.20  # 20 cm, keep in sync with LocalMap.landmark_radius

# 2x2 m local map, centered on mirte.
MAP_XLIM = (-1, 1)
MAP_YLIM = (0, 2)
GRID_STEP = 0.25  # 25 cm


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def plot_local_map(landmarks):
    """Plot authoritative rectangles and their robot-clearance boundaries."""
    plt.clf();ax=plt.gca()
    if hasattr(landmarks,"boxes"):
        from matplotlib.patches import Polygon
        from matplotlib.path import Path as PlotPath
        world_map=landmarks
        world_map.draw_map()
        for i,box in world_map.boxes.items():
            ax.add_patch(Polygon(box["corners"],color="red",alpha=.25))
            # Rounded rectangle inflation via level contour of exact distance.
            center=box["box_center"];r=world_map.mirte_radius+world_map.clearance_margin
            span=np.linalg.norm([box["width"],box["depth"]])/2+r+.05
            xs=np.linspace(center[0]-span,center[0]+span,70);zs=np.linspace(center[1]-span,center[1]+span,70)
            dist=np.array([[world_map.distance_to_box([x,z],i) for x in xs] for z in zs])
            ax.contour(xs,zs,dist,levels=[r],colors="red",linestyles="dashed")
        radius=world_map.mirte_radius
    else:
        radius=MIRTE_RADIUS
        for position,i in landmarks:
            ax.add_patch(Circle(position,LANDMARK_RADIUS,color="red",alpha=.2));ax.text(*position,f"ID {i}")
        ax.set_xlim(*MAP_XLIM);ax.set_ylim(*MAP_YLIM)
    ax.add_patch(Circle((0,0),radius,color="blue",alpha=.3))
    ax.set_xlabel("x (m)");ax.set_ylabel("z (m)");ax.set_aspect("equal");ax.grid(True)
    ax.set_title("Local boxes and robot-clearance boundary")


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

def save_map_json(landmarks,filename):
    if hasattr(landmarks,"boxes"):
        def encode(value):
            if isinstance(value,np.ndarray): return value.tolist()
            if isinstance(value,np.generic): return value.item()
            raise TypeError(type(value).__name__)
        data={"frame":"robot centre at capture; x right, z forward; metres",
              "robot_radius_m":landmarks.mirte_radius,"clearance_margin_m":landmarks.clearance_margin,
              "boxes":landmarks.boxes}
        with open(filename,"w") as f: json.dump(data,f,indent=2,default=encode)
    else:
        data=[{"id":int(i),"x_m":float(p[0]),"z_m":float(p[1])} for p,i in landmarks]
        with open(filename,"w") as f: json.dump(data,f,indent=2)
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

            plot_local_map(local_map)

            time.sleep(0.3)

    finally:

        print("\nSaving final map...")

        save_map_json(
            local_map,
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
