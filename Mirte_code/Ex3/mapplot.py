import cv2
import time
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../Mirte/ku_mirte_python'
    )
)

from ku_mirte import KU_Mirte


def plot_landmark_map_2d(landmark_map):
    plt.clf()

    # Camera position
    plt.scatter(0, 0)
    plt.text(0, 0, "Camera", fontsize=10)

    for item in landmark_map:
        (x, z), landmark_id = item
        plt.scatter(x, z)
        plt.text(x, z, f"ID {landmark_id}", fontsize=10)

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.title("2D Landmark Map")
    plt.grid(True)
    plt.axis("equal")

    plt.xlim(-3000, 3000)
    plt.ylim(0, 5000)

    plt.pause(0.01)


def save_map_json(landmark_map, filename="final_map.json"):
    data = []

    for item in landmark_map:
        (x, z), landmark_id = item
        data.append({
            "id": int(landmark_id),
            "x_mm": float(x),
            "z_mm": float(z)
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved map data to {filename}")


mirte = KU_Mirte()
time.sleep(1)

distortion_coeffs = np.zeros(5)

arucoMarkerLength = 145
f = 609.9
fx = f
fy = f

cx = 640 / 2
cy = 480 / 2

intrinsic_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

plt.ion()

landmark_map = []

try:
    while True:
        landmark_map = []

        img = mirte.get_image_compressed()
        corners, ids, _ = cv2.aruco.detectMarkers(img, arucoDict)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                arucoMarkerLength,
                intrinsic_matrix,
                distortion_coeffs
            )

            id_set = set()

            for i in range(len(ids)):
                landmark_id = int(ids[i][0])

                if landmark_id not in id_set:
                    id_set.add(landmark_id)

                    x, y, z = tvecs[i][0]
                    landmark_map.append([(x, z), landmark_id])

        print(landmark_map)
        plot_landmark_map_2d(landmark_map)

        time.sleep(0.3)

except KeyboardInterrupt:
    print("\nStopping... saving map.")

    save_map_json(landmark_map, "final_map.json")
    plt.savefig("final_map.png")
    print("Saved plot to final_map.png")

finally:
    plt.close("all")
    del mirte