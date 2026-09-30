#!/usr/bin/env python3
"""
angle.py

Simpel ArUco-maaling paa MIRTE.

Bruger samme kamera-metode som jeres gamle fungerende fil:
    mirte.get_image_compressed()

Viser for hver ArUco-kode:
- ID
- afstand til midten af markoeren
- vinkel relativt til kameraets retning

Fortegn:
- positiv vinkel = venstre
- negativ vinkel = hoejre
"""

import cv2
import sys
import os
import math
import time
import numpy as np

# Samme import-metode som i jeres gamle kamera-fil
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../Mirte/ku_mirte_python'
    )
)

from ku_mirte import KU_Mirte


# ------------------------------------------------------------
# Kamera / ArUco-indstillinger
# ------------------------------------------------------------

FOCAL_LENGTH_PX = 609.9

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

cx = IMAGE_WIDTH / 2.0
cy = IMAGE_HEIGHT / 2.0

intrinsic_matrix = np.array([
    [FOCAL_LENGTH_PX, 0, cx],
    [0, FOCAL_LENGTH_PX, cy],
    [0, 0, 1]
], dtype=np.float32)

distortion_coeffs = np.zeros(5)

# Fysisk stoerrelse paa ArUco-markoeren.
# Samme vaerdi som i jeres gamle fil.
ARUCO_MARKER_LENGTH_MM = 145.0

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)


def main():
    print("Starter MIRTE...")
    mirte = KU_Mirte()

    time.sleep(1)

    print("Starter ArUco afstand + vinkel maaling")
    print("Positiv vinkel = venstre")
    print("Negativ vinkel = hoejre")
    print("Tryk Ctrl+C for at stoppe.\n")

    try:
        while True:
            # Samme metode som i jeres gamle fungerende kamera-fil
            img = mirte.get_image_compressed()

            if img is None:
                print("Kunne ikke hente billede.")
                time.sleep(0.1)
                continue

            corners, ids, _ = cv2.aruco.detectMarkers(
                img,
                aruco_dict
            )

            if ids is None or len(ids) == 0:
                print("Ingen ArUco-kode fundet.")
                time.sleep(0.2)
                continue

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                ARUCO_MARKER_LENGTH_MM,
                intrinsic_matrix,
                distortion_coeffs
            )

            for i, marker_id in enumerate(ids.flatten()):
                x, y, z = tvecs[i][0]

                # Vandret afstand fra kamera til centrum af markoeren
                distance_mm = math.sqrt(x*x + z*z)
                distance_m = distance_mm / 1000.0

                # Samme vinkelberegning som i jeres gamle kamera-fil
                angle_rad = -math.atan2(x, z)
                angle_deg = math.degrees(angle_rad)

                print(
                    f"ID {int(marker_id)}: "
                    f"distance = {distance_m:.3f} m, "
                    f"angle = {angle_deg:+.3f} deg, "
                    f"{angle_rad:+.5f} rad"
                )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopper.")

    finally:
        del mirte


if __name__ == "__main__":
    main()
