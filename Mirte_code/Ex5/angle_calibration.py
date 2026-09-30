#!/usr/bin/env python3
"""
angle_calibration.py

Simpel kalibreringstest til MIRTE med den eksisterende Camera-klasse.

Opstilling:
1. Boks direkte foran robotten ved ca. 1.00 m -> forventet vinkel 0 deg
2. Flyt boksen 10 cm til venstre -> forventet ca. +5.71 deg
3. Flyt boksen 10 cm til hoejre  -> forventet ca. -5.71 deg

Fortegn følger camera.py:
    positiv vinkel = venstre
    negativ vinkel = hoejre

Taster:
    0 = center
    l = 10 cm venstre
    r = 10 cm hoejre
    s = gem aktuel maaling i angle_calibration.csv
    q = afslut

VIGTIGT:
Denne fil forventer, at camera.py ligger i samme mappe.
"""

import csv
import math
import os
from datetime import datetime

import cv2

from camera import Camera


FORWARD_DISTANCE_M = 1.00
SIDE_OFFSET_M = 0.10

EXPECTED_SIDE_ANGLE_DEG = math.degrees(
    math.atan2(SIDE_OFFSET_M, FORWARD_DISTANCE_M)
)

CSV_FILE = "angle_calibration.csv"


def expected_angle(stage):
    """
    Fortegn følger camera.py:
      venstre = positiv
      hoejre  = negativ
    """
    if stage == "center":
        return 0.0
    if stage == "left_10cm":
        return EXPECTED_SIDE_ANGLE_DEG
    if stage == "right_10cm":
        return -EXPECTED_SIDE_ANGLE_DEG
    return None


def ensure_csv_header():
    if os.path.exists(CSV_FILE):
        return

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "stage",
            "marker_id",
            "expected_angle_deg",
            "measured_angle_deg",
            "distance_m",
        ])


def save_measurements(stage, measurements):
    ensure_csv_header()

    expected = expected_angle(stage)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        for marker_id, distance_m, angle_deg in measurements:
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                stage,
                marker_id,
                f"{expected:.6f}",
                f"{angle_deg:.6f}",
                f"{distance_m:.6f}",
            ])

            print("\nGemte maaling:")
            print(f"  stage:             {stage}")
            print(f"  marker ID:         {marker_id}")
            print(f"  forventet vinkel:  {expected:+.3f} deg")
            print(f"  maalt vinkel:       {angle_deg:+.3f} deg")
            print(f"  afstand:            {distance_m:.3f} m")
            print(f"  fil:                {CSV_FILE}")


def main():
    print("Starter kamera...")
    cam = Camera(0, robottype="arlo", useCaptureThread=False)

    window_name = "ArUco angle calibration"
    cv2.namedWindow(window_name)

    stage = "center"
    latest_measurements = []

    ensure_csv_header()

    print("\n==============================================")
    print("ArUco angle calibration")
    print("==============================================")
    print(f"Testafstand:  {FORWARD_DISTANCE_M:.2f} m")
    print(f"Sideflyt:     {SIDE_OFFSET_M:.2f} m")
    print(
        f"Forventet sidevinkel: "
        f"{EXPECTED_SIDE_ANGLE_DEG:.3f} deg"
    )
    print("")
    print("Fortegn:")
    print("  venstre = positiv")
    print("  hoejre  = negativ")
    print("")
    print("Taster:")
    print("  0 = center")
    print("  l = 10 cm venstre")
    print("  r = 10 cm hoejre")
    print("  s = gem maaling")
    print("  q = afslut")
    print("==============================================\n")

    try:
        while True:
            frame = cam.get_next_frame()

            ids, dists_cm, angles_rad = cam.detect_aruco_objects(frame)

            latest_measurements = []

            if ids is not None:
                for i in range(len(ids)):
                    marker_id = int(ids[i])
                    distance_m = float(dists_cm[i]) / 100.0
                    angle_deg = math.degrees(float(angles_rad[i]))

                    latest_measurements.append(
                        (marker_id, distance_m, angle_deg)
                    )

                    print(
                        f"ID {marker_id}: "
                        f"distance = {distance_m:.3f} m, "
                        f"angle = {angle_deg:+.3f} deg"
                    )

                cam.draw_aruco_objects(frame)

            expected = expected_angle(stage)

            cv2.putText(
                frame,
                f"Stage: {stage} | expected: {expected:+.2f} deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "0=center  l=left  r=right  s=save  q=quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("0"):
                stage = "center"
                print("\nStage = center")
                print("Forventet vinkel = 0.000 deg\n")

            elif key == ord("l"):
                stage = "left_10cm"
                print("\nStage = 10 cm venstre")
                print(
                    f"Forventet vinkel = "
                    f"+{EXPECTED_SIDE_ANGLE_DEG:.3f} deg\n"
                )

            elif key == ord("r"):
                stage = "right_10cm"
                print("\nStage = 10 cm hoejre")
                print(
                    f"Forventet vinkel = "
                    f"-{EXPECTED_SIDE_ANGLE_DEG:.3f} deg\n"
                )

            elif key == ord("s"):
                if not latest_measurements:
                    print("\nIngen ArUco-kode fundet - intet gemt.\n")
                else:
                    save_measurements(stage, latest_measurements)

    finally:
        cv2.destroyAllWindows()
        cam.terminateCaptureThread()
        del cam


if __name__ == "__main__":
    main()
