#Mirte Proto code

import time
import sys
import os
import json
import signal
import matplotlib.pyplot as plt

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


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def plot_local_map(landmarks):
    plt.clf()

    # Mirte position
    plt.scatter(0, 0)
    plt.text(0, 0, "Mirte", fontsize=10)

    # Landmarks
    for position, landmark_id in landmarks:
        x, z = position

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
    plt.grid(True)

    # Fixed local map area
    plt.xlim(-3, 3)
    plt.ylim(0, 5)

    # Equal geometric scale
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

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


# --------------------------------------------------
# Start MIRTE
# --------------------------------------------------

mirte = KU_Mirte()
time.sleep(1)

local_map = LocalMap()


# --------------------------------------------------
# Ctrl+C handling
# --------------------------------------------------

running = True


def handle_sigint(signum, frame):
    global running

    print(
        "\nCtrl+C received. "
        "Finishing current iteration and saving map..."
    )

    running = False


signal.signal(signal.SIGINT, handle_sigint)


# --------------------------------------------------
# Interactive plot
# --------------------------------------------------

plt.ion()


# --------------------------------------------------
# Main loop
# --------------------------------------------------

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
