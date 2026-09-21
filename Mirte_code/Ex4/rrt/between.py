"""
drive_between_aruco.py

Finder to ArUco-landmarks med koden fra local_map.py og kører MIRTE
mod midtpunktet mellem dem.

Filen er lavet til at ligge i samme mappe som:
    local_map.py

Den bruger local_map.LocalMap som "library", så vi genbruger jeres
kamera-kalibrering, ArUco-dictionary, markerstørrelse og camera offset.

Strategi:
1. Find mindst to ArUco-markers.
2. Vælg et par og lås deres IDs.
3. Beregn midtpunktet mellem de to landmarks i robot-koordinater.
4. Drej mod midtpunktet.
5. Kør et lille stykke frem.
6. Tag et nyt billede og korriger igen.
7. Stop når robotten er tæt på midtpunktet.

Det er bevidst lavet som små bevægelser, så motorfejl ikke akkumulerer
lige så meget som ved én stor åben-loop bevægelse.
"""

import math
import os
import sys
import time

import numpy as np


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
# Indstillinger
# ---------------------------------------------------------------------------

# Hvis I KENDER de to ArUco IDs, så skriv dem her, fx:
# TARGET_IDS = (4, 7)
#
# Hvis None vælges automatisk de to synlige markers med størst
# vandret afstand mellem sig. Derefter låses de IDs resten af kørslen.
TARGET_IDS = None

# Hvor tæt robot-centret skal være på midtpunktet før vi stopper.
STOP_DISTANCE = 0.12       # meter

# Små skridt gør kørslen mere robust over for motorfejl.
MAX_FORWARD_STEP = 0.12    # meter pr. iteration

LINEAR_SPEED = 0.15        # m/s
ANGULAR_SPEED = 0.35       # rad/s

# Hvis vinkelfejlen er større end dette, drejer vi først uden at køre frem.
ANGLE_TOLERANCE = math.radians(5.0)

# Søgning når begge markers ikke kan ses.
SEARCH_TURN_ANGLE = math.radians(10.0)
SEARCH_PAUSE = 0.20

# Maksimum antal loop-iterationer som ekstra sikkerhed.
MAX_ITERATIONS = 100

# Pause efter en fysisk bevægelse før næste kameramåling.
CAMERA_SETTLE_TIME = 0.25


# ---------------------------------------------------------------------------
# Hjælpefunktioner
# ---------------------------------------------------------------------------

def stop_robot(mirte):
    """Forsøg at sende en stopkommando til robotten."""
    try:
        mirte.drive(0.0, 0.0, 0.1)
    except Exception:
        pass


def landmark_dict(landmarks):
    """
    Konverterer LocalMap-formatet:
        [[np.array([x, z]), id], ...]
    til:
        {id: np.array([x, z]), ...}

    x = sideværts position set fra robotten
        negativ = venstre
        positiv = højre

    z = fremad fra robotten
        positiv = foran robotten
    """
    result = {}

    for position, marker_id in landmarks:
        result[int(marker_id)] = np.asarray(position, dtype=float)

    return result


def print_landmarks(points):
    if not points:
        print("Ingen ArUco markers fundet.")
        return

    print("Synlige ArUco markers:")
    for marker_id in sorted(points):
        x, z = points[marker_id]
        print(
            f"  ID {marker_id:3d}: "
            f"x = {x:+.3f} m, z = {z:+.3f} m"
        )


def choose_marker_pair(points):
    """
    Vælger hvilke to markers der skal bruges.

    Hvis TARGET_IDS er sat, bruges præcis de IDs.

    Ellers vælges de to markers med størst forskel i x-retningen.
    Det giver god mening når to landmarks står på hver sin side af
    den passage robotten skal køre imellem.
    """

    if TARGET_IDS is not None:
        id_a, id_b = TARGET_IDS

        if id_a in points and id_b in points:
            return int(id_a), int(id_b)

        return None

    ids = list(points.keys())

    if len(ids) < 2:
        return None

    best_pair = None
    best_horizontal_distance = -1.0

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a = ids[i]
            id_b = ids[j]

            horizontal_distance = abs(
                points[id_a][0] - points[id_b][0]
            )

            if horizontal_distance > best_horizontal_distance:
                best_horizontal_distance = horizontal_distance
                best_pair = (id_a, id_b)

    return best_pair


def midpoint_between(points, marker_pair):
    id_a, id_b = marker_pair
    p_a = points[id_a]
    p_b = points[id_b]

    midpoint = (p_a + p_b) / 2.0
    return midpoint, p_a, p_b


def midpoint_geometry(midpoint):
    """
    LocalMap bruger [x, z]:
        x = sideværts
        z = fremad

    Afstanden er derfor:
        sqrt(x^2 + z^2)

    Robot-yaw følger normal ROS-konvention:
        positiv yaw = venstre / mod uret

    Kameraets x er positiv mod højre, så vinklen får et minus:
        theta = -atan2(x, z)
    """
    x = float(midpoint[0])
    z = float(midpoint[1])

    distance = math.hypot(x, z)
    turn_angle = -math.atan2(x, z)

    return distance, turn_angle


def rotate_robot(mirte, angle):
    """Drej robotten angle radianer ved fast angular speed."""
    if abs(angle) < 1e-6:
        return

    angular_velocity = math.copysign(ANGULAR_SPEED, angle)
    duration = abs(angle) / ANGULAR_SPEED

    print(
        f"Drejer {math.degrees(angle):+.1f} grader "
        f"({duration:.2f} s)"
    )

    mirte.drive(
        0.0,
        angular_velocity,
        duration
    )


def drive_forward(mirte, distance):
    """Kør et kontrolleret lille stykke lige frem."""
    if distance <= 0.0:
        return

    duration = distance / LINEAR_SPEED

    print(
        f"Kører {distance:.3f} m frem "
        f"({duration:.2f} s)"
    )

    mirte.drive(
        LINEAR_SPEED,
        0.0,
        duration
    )


def search_for_pair(mirte):
    """
    Drej lidt til venstre for at søge efter markers.
    Hvis robotten søger den forkerte vej i jeres opsætning, kan fortegnet
    på SEARCH_TURN_ANGLE ændres.
    """
    print("Kan ikke se begge target-markers. Søger...")

    rotate_robot(
        mirte,
        SEARCH_TURN_ANGLE
    )

    time.sleep(SEARCH_PAUSE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mirte = None

    try:
        print("Starter MIRTE...")
        mirte = KU_Mirte()

        # Giv kamera/ROS lidt tid til at starte.
        time.sleep(1.0)

        # Vi bruger jeres eksisterende ArUco-kode fra local_map.py.
        local_map = LocalMap()

        locked_pair = None

        for iteration in range(1, MAX_ITERATIONS + 1):
            print()
            print("=" * 64)
            print(f"Iteration {iteration}/{MAX_ITERATIONS}")

            # LocalMap.update() kalder get_map_from_mirte(), som:
            # - tager kamera-billede
            # - finder ArUco markers
            # - estimatePoseSingleMarkers()
            # - korrigerer camera offset
            local_map.update(mirte)

            points = landmark_dict(local_map.landmarks)
            print_landmarks(points)

            # Første gang vi ser et brugbart marker-par, låser vi deres IDs.
            if locked_pair is None:
                locked_pair = choose_marker_pair(points)

                if locked_pair is not None:
                    print(
                        f"Låser target-markers: "
                        f"ID {locked_pair[0]} og ID {locked_pair[1]}"
                    )

            # Hvis vi endnu ikke har fundet to markers.
            if locked_pair is None:
                search_for_pair(mirte)
                continue

            id_a, id_b = locked_pair

            # Hvis én af de låste markers midlertidigt forsvinder ud af billedet.
            if id_a not in points or id_b not in points:
                print(
                    f"Mangler ID {id_a} eller ID {id_b} i kameraet."
                )
                search_for_pair(mirte)
                continue

            midpoint, p_a, p_b = midpoint_between(
                points,
                locked_pair
            )

            distance, turn_angle = midpoint_geometry(midpoint)

            gap_width = float(np.linalg.norm(p_a - p_b))

            print(
                f"Marker ID {id_a}: "
                f"[x={p_a[0]:+.3f}, z={p_a[1]:+.3f}] m"
            )
            print(
                f"Marker ID {id_b}: "
                f"[x={p_b[0]:+.3f}, z={p_b[1]:+.3f}] m"
            )
            print(f"Afstand mellem markers: {gap_width:.3f} m")
            print(
                "Midtpunkt: "
                f"[x={midpoint[0]:+.3f}, z={midpoint[1]:+.3f}] m"
            )
            print(f"Afstand til midtpunkt: {distance:.3f} m")
            print(
                f"Vinkel til midtpunkt: "
                f"{math.degrees(turn_angle):+.1f} grader"
            )

            # Målet er nået.
            if distance <= STOP_DISTANCE:
                stop_robot(mirte)

                print()
                print("MÅL NÅET")
                print(
                    f"Robotten er inden for {STOP_DISTANCE:.2f} m "
                    "af midtpunktet mellem de to ArUco markers."
                )
                return

            # Hvis midtpunktet faktisk ligger bag robotten, er noget gået galt
            # eller robotten er kørt forbi. Drej først i stedet for at bakke.
            if midpoint[1] <= 0.0:
                print(
                    "Midtpunktet ligger ikke foran robotten. "
                    "Drejer mod det før videre kørsel."
                )
                rotate_robot(mirte, turn_angle)
                time.sleep(CAMERA_SETTLE_TIME)
                continue

            # Først ret retningen ind.
            if abs(turn_angle) > ANGLE_TOLERANCE:
                rotate_robot(mirte, turn_angle)
                time.sleep(CAMERA_SETTLE_TIME)
                continue

            # Derefter kør kun et lille stykke og mål igen.
            remaining = max(0.0, distance - STOP_DISTANCE)
            forward_step = min(
                MAX_FORWARD_STEP,
                remaining
            )

            drive_forward(
                mirte,
                forward_step
            )

            time.sleep(CAMERA_SETTLE_TIME)

        print()
        print(
            "STOP: Maksimum antal iterationer blev nået "
            "uden at nå midtpunktet."
        )

    except KeyboardInterrupt:
        print()
        print("Ctrl+C modtaget. Stopper robotten.")

    except Exception as exc:
        print()
        print("FEJL:")
        print(exc)
        raise

    finally:
        if mirte is not None:
            stop_robot(mirte)

        print("Robot stoppet.")


if __name__ == "__main__":
    main()
