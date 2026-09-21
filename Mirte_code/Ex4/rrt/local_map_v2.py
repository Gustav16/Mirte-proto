# local_map_v2.py
#
# Local landmark map for MIRTE.
#
# Main change compared with the current local_map.py:
# The ArUco measurement is transformed from the camera origin to the
# ROBOT CENTER. If the camera is mounted CAMERA_FORWARD_OFFSET metres
# in front of the robot center, a landmark in front of the camera is
# CAMERA_FORWARD_OFFSET farther away from the robot center:
#
#     p_robot = p_camera + p_camera_origin_in_robot
#
# Therefore the camera offset is ADDED, not subtracted.

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../Mirte/ku_mirte_python"
    )
)

from ku_mirte import KU_Mirte


# ---------------------------------------------------------------------------
# Camera / ArUco calibration
# ---------------------------------------------------------------------------

distortion_coeffs = np.zeros(5)

ARUCO_MARKER_LENGTH_MM = 145

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

arucoDict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)


# IMPORTANT:
# Measure this on the physical robot.
#
# Your previous local_map.py used 0.10 m, although its comment said 8 cm.
# For the first test this is therefore kept at 0.10 m.
CAMERA_FORWARD_OFFSET = 0.10

# Camera origin position expressed in the robot-center frame.
# Map convention:
#   x = right
#   z = forward
CAMERA_OFFSET_FROM_ROBOT_CENTER = np.array([
    0.0,
    CAMERA_FORWARD_OFFSET
], dtype=float)


# Approximate depth of the ArUco box.
#
# IMPORTANT UNIT DETAIL:
# estimatePoseSingleMarkers() returns tvec in the same unit as the marker
# length. Since ARUCO_MARKER_LENGTH_MM = 145, tvec is in millimetres.
# The box-depth offset must therefore ALSO be in millimetres here.
OBJECT_DEPTH_M = 0.20
OBJECT_DEPTH_MM = OBJECT_DEPTH_M * 1000.0

# Shift from the ArUco paper plane toward the approximate box center.
obstacle_offset_3d_mm = np.array([
    0.0,
    0.0,
    OBJECT_DEPTH_MM / 2.0
])


class LocalMap:
    """
    Local 2D landmark map in the ROBOT-CENTER frame.

    A landmark is stored as:

        [np.array([x, z]), landmark_id]

    where:
        x < 0 : left of MIRTE
        x > 0 : right of MIRTE
        z > 0 : in front of MIRTE
    """

    def __init__(
        self,
        landmarks=None,
        landmark_radius=0.30,
        mirte_radius=0.20,
        camera_offset=CAMERA_OFFSET_FROM_ROBOT_CENTER,
        low=(-2.0, 0.0),
        high=(2.0, 3.0),
        res=0.10
    ):
        self.landmarks = [] if landmarks is None else landmarks

        self.mirte_radius = float(mirte_radius)
        self.landmark_radius = float(landmark_radius)

        self.camera_offset = np.asarray(
            camera_offset,
            dtype=float
        )

        self.map_area = [
            np.asarray(low, dtype=float),
            np.asarray(high, dtype=float)
        ]

        self.resolution = float(res)

    def update(self, mirte):
        self.landmarks = self.get_map_from_mirte(mirte)

    def in_collision(self, pos):
        """
        Collision check using circular landmark and MIRTE approximations.

        A point is unsafe if the robot center would be closer than

            landmark_radius + mirte_radius

        to a landmark center.
        """
        pos = np.asarray(pos, dtype=float)

        # Treat leaving the planned map area as collision.
        if np.any(pos < self.map_area[0]) or np.any(pos > self.map_area[1]):
            return 1

        clearance = self.landmark_radius + self.mirte_radius

        for landmark, landmark_id in self.landmarks:
            landmark = np.asarray(landmark, dtype=float)

            if np.linalg.norm(pos - landmark) <= clearance:
                return 1

        return 0

    def get_map_from_mirte(self, mirte):
        """
        Take one camera frame and detect all unique ArUco landmarks.

        Returned positions are expressed relative to the ROBOT CENTER.
        """
        landmark_map = []

        img = mirte.get_image_compressed()

        if img is None:
            return landmark_map

        corners, ids, _ = cv2.aruco.detectMarkers(
            img,
            arucoDict
        )

        if ids is None:
            return landmark_map

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            ARUCO_MARKER_LENGTH_MM,
            intrinsic_matrix,
            distortion_coeffs
        )

        seen_ids = set()

        for i in range(len(ids)):
            landmark_id = int(ids[i][0])

            if landmark_id in seen_ids:
                continue

            seen_ids.add(landmark_id)

            # Rotation of the marker relative to the camera.
            R, _ = cv2.Rodrigues(rvecs[i][0])

            # Estimate approximate center of the physical box.
            center_camera_mm = (
                tvecs[i][0]
                + R @ obstacle_offset_3d_mm
            )

            x_mm, y_mm, z_mm = center_camera_mm

            # Camera-frame 2D floor position in metres.
            landmark_camera = np.array([
                x_mm / 1000.0,
                z_mm / 1000.0
            ])

            # Camera -> robot-center transform.
            #
            # Camera is CAMERA_FORWARD_OFFSET in front of robot center,
            # therefore a landmark in front of the camera is that much
            # farther from the robot center.
            landmark_robot = (
                landmark_camera
                + self.camera_offset
            )

            landmark_map.append([
                landmark_robot,
                landmark_id
            ])

        return landmark_map

    def draw_map(self):
        """
        Basic plot used by RRT/debug code.
        """
        ax = plt.gca()

        clearance = (
            self.landmark_radius
            + self.mirte_radius
        )

        for landmark, landmark_id in self.landmarks:
            x, z = landmark

            ax.add_patch(
                Circle(
                    (x, z),
                    clearance,
                    fill=False
                )
            )

            ax.scatter(x, z)
            ax.text(
                x + 0.02,
                z + 0.02,
                f"ID {landmark_id}"
            )

        ax.scatter(0.0, 0.0, marker="^")

        ax.set_xlim(
            self.map_area[0][0],
            self.map_area[1][0]
        )

        ax.set_ylim(
            self.map_area[0][1],
            self.map_area[1][1]
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        ax.grid(True)
