#!/usr/bin/env python3
"""
angle_calibration.py

Kalibrering af ArUco-vinkel på MIRTE.

Ide:
- Vi bruger jeres tidligere kalibrerede focal length:
      f = 609.9234 px
- En ArUco-markør direkte foran kameraets center bør give ca. 0 grader.
- Hvis boksen står 1.00 m fremme og flyttes 0.10 m til siden,
  er den geometrisk forventede vinkel:

      theta = atan2(0.10, 1.00) = 5.7106 grader

Programmet finder centrum af ArUco-markøren og beregner:

      theta = atan2(u - cx, f)

hvor
    u  = markørens center-x i billedet
    cx = billedets center-x
    f  = focal length i pixels

Taster i live-vinduet:
    0 = markøren står i midten (forventet 0 grader)
    l = markøren står 10 cm til venstre ved 1 m
    r = markøren står 10 cm til højre ved 1 m
    s = gem den aktuelle måling i CSV
    q = afslut

Eksempler:
    python3 angle_calibration.py
    python3 angle_calibration.py --camera 0
    python3 angle_calibration.py --image test.png
"""

import argparse
import csv
import math
import os
from datetime import datetime

import cv2
import numpy as np


# -------------------------------------------------------------------------
# Jeres tidligere kamerakalibrering
# -------------------------------------------------------------------------

FOCAL_LENGTH_PX = 609.9234
FOCAL_LENGTH_STD_PX = 4.831246

# Fysisk test-opstilling
FORWARD_DISTANCE_M = 1.00
SIDE_OFFSET_M = 0.10

EXPECTED_SIDE_ANGLE_DEG = math.degrees(
    math.atan2(SIDE_OFFSET_M, FORWARD_DISTANCE_M)
)

CSV_FILE = "angle_calibration.csv"


# -------------------------------------------------------------------------
# ArUco
# -------------------------------------------------------------------------

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

try:
    ARUCO_PARAMS = cv2.aruco.DetectorParameters()
    ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
except AttributeError:
    # Ældre OpenCV-version
    ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
    ARUCO_DETECTOR = None


def detect_markers(frame):
    """Returnerer (corners, ids) for alle ArUco-markører i billedet."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ARUCO_DETECTOR is not None:
        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            ARUCO_DICT,
            parameters=ARUCO_PARAMS
        )

    return corners, ids


def marker_center(corner):
    """
    Beregner centrum af én ArUco-markør.

    corner har normalt formen (1, 4, 2).
    """
    pts = corner.reshape(4, 2)
    center = pts.mean(axis=0)
    return float(center[0]), float(center[1])


def marker_pixel_height(corner):
    """
    Estimerer markørens højde i pixels som gennemsnittet
    af venstre og højre side.
    """
    pts = corner.reshape(4, 2)

    top_left, top_right, bottom_right, bottom_left = pts

    left_height = np.linalg.norm(bottom_left - top_left)
    right_height = np.linalg.norm(bottom_right - top_right)

    return float((left_height + right_height) / 2.0)


def angle_from_pixel(center_x, image_width, focal_length_px):
    """
    Beregner bearing/vinkel til markørens centrum.

    Positiv = højre i billedet.
    Negativ = venstre i billedet.
    """
    cx = image_width / 2.0
    pixel_offset = center_x - cx

    angle_rad = math.atan2(pixel_offset, focal_length_px)
    angle_deg = math.degrees(angle_rad)

    return angle_deg, pixel_offset, cx


def estimated_distance_from_marker_height(
    marker_height_px,
    marker_height_m,
    focal_length_px
):
    """
    Simpel pinhole-distance:
        Z = f * X / x

    Kun gyldig hvis marker_height_m svarer til den fysiske størrelse,
    som focal length-kalibreringen blev lavet med.
    """
    if marker_height_px <= 0:
        return None

    return focal_length_px * marker_height_m / marker_height_px


def expected_angle_for_stage(stage):
    if stage == "center":
        return 0.0
    elif stage == "left_10cm":
        return -EXPECTED_SIDE_ANGLE_DEG
    elif stage == "right_10cm":
        return EXPECTED_SIDE_ANGLE_DEG
    return None


def ensure_csv_header(path):
    if os.path.exists(path):
        return

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "stage",
            "marker_id",
            "expected_angle_deg",
            "measured_angle_deg",
            "error_deg",
            "marker_center_x_px",
            "image_center_x_px",
            "pixel_offset_px",
            "marker_height_px",
            "estimated_distance_m",
            "focal_length_px",
        ])


def save_measurement(
    path,
    stage,
    marker_id,
    expected_angle,
    measured_angle,
    marker_center_x,
    image_center_x,
    pixel_offset,
    marker_height_px,
    estimated_distance_m
):
    ensure_csv_header(path)

    error = None
    if expected_angle is not None:
        error = measured_angle - expected_angle

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            stage,
            marker_id,
            "" if expected_angle is None else f"{expected_angle:.6f}",
            f"{measured_angle:.6f}",
            "" if error is None else f"{error:.6f}",
            f"{marker_center_x:.3f}",
            f"{image_center_x:.3f}",
            f"{pixel_offset:.3f}",
            f"{marker_height_px:.3f}",
            "" if estimated_distance_m is None else f"{estimated_distance_m:.6f}",
            f"{FOCAL_LENGTH_PX:.6f}",
        ])

    print("\nGemte måling:")
    print(f"  stage:              {stage}")
    print(f"  marker ID:          {marker_id}")
    if expected_angle is not None:
        print(f"  forventet vinkel:   {expected_angle:+.3f} deg")
    print(f"  målt vinkel:        {measured_angle:+.3f} deg")
    if error is not None:
        print(f"  fejl:                {error:+.3f} deg")
    print(f"  pixel offset:        {pixel_offset:+.1f} px")
    if estimated_distance_m is not None:
        print(f"  estimeret afstand:   {estimated_distance_m:.3f} m")
    print(f"  CSV:                 {path}\n")


def process_frame(frame, marker_height_m=None):
    """
    Finder alle markører og returnerer målinger.
    """
    corners, ids = detect_markers(frame)

    measurements = []

    if ids is None or len(ids) == 0:
        return measurements

    h, w = frame.shape[:2]

    for corner, marker_id in zip(corners, ids.flatten()):
        mx, my = marker_center(corner)

        measured_angle_deg, pixel_offset, image_center_x = angle_from_pixel(
            mx,
            w,
            FOCAL_LENGTH_PX
        )

        height_px = marker_pixel_height(corner)

        distance_m = None
        if marker_height_m is not None:
            distance_m = estimated_distance_from_marker_height(
                height_px,
                marker_height_m,
                FOCAL_LENGTH_PX
            )

        measurements.append({
            "id": int(marker_id),
            "corner": corner,
            "center_x": mx,
            "center_y": my,
            "angle_deg": measured_angle_deg,
            "pixel_offset": pixel_offset,
            "image_center_x": image_center_x,
            "marker_height_px": height_px,
            "distance_m": distance_m,
        })

    return measurements


def draw_measurements(frame, measurements, stage):
    h, w = frame.shape[:2]

    # Kameraets centerlinje
    cx = int(w / 2)
    cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1)

    expected = expected_angle_for_stage(stage)

    for m in measurements:
        pts = m["corner"].reshape(4, 2).astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        mx = int(round(m["center_x"]))
        my = int(round(m["center_y"]))

        cv2.circle(frame, (mx, my), 5, (0, 0, 255), -1)

        label = (
            f"ID {m['id']}  angle={m['angle_deg']:+.2f} deg  "
            f"dx={m['pixel_offset']:+.1f}px"
        )

        if m["distance_m"] is not None:
            label += f"  Z~{m['distance_m']:.2f}m"

        cv2.putText(
            frame,
            label,
            (max(5, mx - 180), max(25, my - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    stage_text = f"Stage: {stage}"
    if expected is not None:
        stage_text += f" | expected: {expected:+.3f} deg"

    cv2.putText(
        frame,
        stage_text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "0=center  l=left 10cm  r=right 10cm  s=save  q=quit",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def run_on_image(image_path, marker_height_m=None):
    frame = cv2.imread(image_path)

    if frame is None:
        raise RuntimeError(f"Kunne ikke laese billedet: {image_path}")

    measurements = process_frame(frame, marker_height_m)

    print(f"\nBillede: {image_path}")
    print(f"Focal length: {FOCAL_LENGTH_PX:.4f} px")

    if not measurements:
        print("Ingen ArUco-markoerer fundet.")
        return

    for m in measurements:
        print(
            f"ID {m['id']}: "
            f"angle={m['angle_deg']:+.3f} deg, "
            f"pixel_offset={m['pixel_offset']:+.1f}px"
        )

    frame = draw_measurements(frame, measurements, "unknown")
    output_path = "angle_calibration_result.png"
    cv2.imwrite(output_path, frame)
    print(f"Resultat gemt som: {output_path}")


def run_live(camera_index, marker_height_m=None):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Kunne ikke aabne kamera index {camera_index}.\n"
            "Hvis MIRTE-kameraet kun er tilgaengeligt via ROS2, "
            "skal frame-inputtet kobles til jeres eksisterende kamera-kode."
        )

    stage = "center"

    print("\n========================================================")
    print("ArUco angle calibration")
    print("========================================================")
    print(f"Focal length: {FOCAL_LENGTH_PX:.4f} px")
    print(f"Std(f):       {FOCAL_LENGTH_STD_PX:.4f} px")
    print(f"Testafstand:  {FORWARD_DISTANCE_M:.2f} m")
    print(f"Sideflyt:     {SIDE_OFFSET_M:.2f} m")
    print(
        f"Forventet vinkel ved +/-10 cm og 1 m: "
        f"+/-{EXPECTED_SIDE_ANGLE_DEG:.4f} deg"
    )
    print()
    print("Taster:")
    print("  0 = center")
    print("  l = 10 cm venstre")
    print("  r = 10 cm hoejre")
    print("  s = gem maaling")
    print("  q = afslut")
    print("========================================================\n")

    ensure_csv_header(CSV_FILE)

    latest_measurements = []

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            print("Kunne ikke hente frame.")
            continue

        latest_measurements = process_frame(frame, marker_height_m)

        display = frame.copy()
        display = draw_measurements(display, latest_measurements, stage)

        cv2.imshow("ArUco angle calibration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("0"):
            stage = "center"
            print("\nStage = center, forventet 0.000 deg")

        elif key == ord("l"):
            stage = "left_10cm"
            print(
                f"\nStage = left_10cm, forventet "
                f"{-EXPECTED_SIDE_ANGLE_DEG:.3f} deg"
            )

        elif key == ord("r"):
            stage = "right_10cm"
            print(
                f"\nStage = right_10cm, forventet "
                f"{EXPECTED_SIDE_ANGLE_DEG:.3f} deg"
            )

        elif key == ord("s"):
            if not latest_measurements:
                print("\nIngen ArUco-markoer fundet - intet gemt.")
                continue

            # Hvis flere markører ses, gemmes alle.
            expected = expected_angle_for_stage(stage)

            for m in latest_measurements:
                save_measurement(
                    CSV_FILE,
                    stage,
                    m["id"],
                    expected,
                    m["angle_deg"],
                    m["center_x"],
                    m["image_center_x"],
                    m["pixel_offset"],
                    m["marker_height_px"],
                    m["distance_m"],
                )

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Test paa et gemt billede i stedet for live kamera.",
    )

    parser.add_argument(
        "--marker-height-mm",
        type=float,
        default=None,
        help=(
            "Fysisk hoejde i mm paa den samme genstand/markoer, "
            "som bruges til afstandsestimat. "
            "Fx 145 hvis det er den korrekte fysiske hoejde."
        ),
    )

    args = parser.parse_args()

    marker_height_m = None
    if args.marker_height_mm is not None:
        marker_height_m = args.marker_height_mm / 1000.0

    if args.image is not None:
        run_on_image(args.image, marker_height_m)
    else:
        run_live(args.camera, marker_height_m)


if __name__ == "__main__":
    main()
