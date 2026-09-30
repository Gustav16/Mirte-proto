#!/usr/bin/env python3
"""
angle.py

Simpel MIRTE ArUco-test:
- henter billeder med den eksisterende Camera-klasse
- finder ArUco-koder
- viser afstand og vinkel

VIGTIGT:
Denne fil forventer, at camera.py ligger i samme mappe.

Camera.detect_aruco_objects() returnerer:
    ids
    distance i cm
    angle i radianer

Vi konverterer her til:
    distance i meter
    angle i grader

Fortegn følger camera.py:
    positiv vinkel = venstre
    negativ vinkel = højre
"""

import math
import cv2

from camera import Camera


def main():
    print("Starter kamera...")
    cam = Camera(0, robottype="arlo", useCaptureThread=False)

    window_name = "ArUco distance and angle"
    cv2.namedWindow(window_name)

    print("\nStarter ArUco-maaling")
    print("Positiv vinkel = venstre")
    print("Negativ vinkel = hoejre")
    print("Tryk q i kameravinduet for at stoppe.\n")

    try:
        while True:
            # Hent billede med den samme Camera-klasse,
            # som I tidligere har brugt på MIRTE.
            frame = cam.get_next_frame()

            # Camera-klassen finder selv ArUco og beregner
            # afstand (cm) og vinkel (radianer).
            ids, dists_cm, angles_rad = cam.detect_aruco_objects(frame)

            if ids is not None:
                for i in range(len(ids)):
                    marker_id = int(ids[i])
                    distance_m = float(dists_cm[i]) / 100.0
                    angle_deg = math.degrees(float(angles_rad[i]))

                    print(
                        f"ID {marker_id}: "
                        f"distance = {distance_m:.3f} m, "
                        f"angle = {angle_deg:+.3f} deg"
                    )

                # Tegn de detekterede ArUco-koder og akser
                cam.draw_aruco_objects(frame)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        cam.terminateCaptureThread()
        del cam


if __name__ == "__main__":
    main()
