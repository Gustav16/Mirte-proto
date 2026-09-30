#!/usr/bin/env python3
"""
simple_aruco_distance_angle.py

Simpel ArUco-måling:
- finder ArUco-markører
- måler afstand
- måler vinkel til markøren

Ingen fejlkorrektion, standardafvigelser eller kalibrerings-CSV.

Formler:
    distance = f * X / x
    angle    = atan2(pixel_offset, f)

hvor:
    f = focal length i pixels
    X = fysisk størrelse af ArUco-markøren
    x = målt størrelse af ArUco-markøren i pixels
"""

import cv2
import numpy as np
import math


# ------------------------------------------------------------
# KONSTANTER
# ------------------------------------------------------------

# Jeres tidligere målte focal length
FOCAL_LENGTH_PX = 609.9234

# Fysisk højde/bredde på den firkant, som I måler i billedet.
# Sæt denne til den rigtige størrelse på selve ArUco-koden.
MARKER_SIZE_MM = 145.0

# Kamera-index
CAMERA_INDEX = 0


# ------------------------------------------------------------
# ARUCO SETUP
# ------------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

try:
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
except AttributeError:
    parameters = cv2.aruco.DetectorParameters_create()
    detector = None


def detect_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=parameters
        )

    return corners, ids


def marker_center(corner):
    pts = corner.reshape(4, 2)

    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])

    return float(center_x), float(center_y)


def marker_size_pixels(corner):
    """
    Bruger gennemsnittet af venstre og højre side
    som markørens højde i pixels.
    """
    pts = corner.reshape(4, 2)

    top_left = pts[0]
    top_right = pts[1]
    bottom_right = pts[2]
    bottom_left = pts[3]

    left_side = np.linalg.norm(bottom_left - top_left)
    right_side = np.linalg.norm(bottom_right - top_right)

    return float((left_side + right_side) / 2.0)


def calculate_distance(marker_pixels):
    """
    Z = f * X / x
    """
    if marker_pixels <= 0:
        return None

    distance_mm = (
        FOCAL_LENGTH_PX
        * MARKER_SIZE_MM
        / marker_pixels
    )

    return distance_mm / 1000.0


def calculate_angle(marker_center_x, image_width):
    """
    Vinkel relativt til kameraets centerlinje.

    Negativ = venstre
    Positiv = højre
    """
    image_center_x = image_width / 2.0

    pixel_offset = marker_center_x - image_center_x

    angle_rad = math.atan2(
        pixel_offset,
        FOCAL_LENGTH_PX
    )

    return math.degrees(angle_rad)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Kunne ikke åbne kameraet.")
        return

    print("Starter ArUco distance + angle måling")
    print("Tryk q for at stoppe.\n")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Kunne ikke hente billede fra kameraet.")
            continue

        corners, ids = detect_markers(frame)

        height, width = frame.shape[:2]

        # Tegn kameraets centerlinje
        center_x = int(width / 2)
        cv2.line(
            frame,
            (center_x, 0),
            (center_x, height),
            (255, 255, 255),
            1
        )

        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):

                # Centrum af ArUco
                mx, my = marker_center(corner)

                # Størrelse i pixels
                size_px = marker_size_pixels(corner)

                # Afstand
                distance_m = calculate_distance(size_px)

                # Vinkel
                angle_deg = calculate_angle(mx, width)

                # Tegn markøren
                pts = corner.reshape(4, 2).astype(int)

                cv2.polylines(
                    frame,
                    [pts],
                    True,
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (int(mx), int(my)),
                    5,
                    (0, 0, 255),
                    -1
                )

                # Tekst i billedet
                text = (
                    f"ID {marker_id} | "
                    f"Distance: {distance_m:.2f} m | "
                    f"Angle: {angle_deg:+.2f} deg"
                )

                cv2.putText(
                    frame,
                    text,
                    (int(mx) - 150, int(my) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                # Samme information i terminalen
                print(
                    f"ID {marker_id}: "
                    f"distance = {distance_m:.3f} m, "
                    f"angle = {angle_deg:+.3f} deg"
                )

        cv2.imshow(
            "ArUco distance and angle",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
