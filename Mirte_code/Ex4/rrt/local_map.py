# local_map.py
#
# Local landmark map for MIRTE.
#
# ArUco tvec is used directly for the landmark position.
# We deliberately do NOT rotate a box-depth offset using rvec, because
# differences/noise in the estimated marker orientation can move otherwise
# aligned boxes differently in both x and z.
#
# The camera position is then translated to the ROBOT CENTER frame.

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


# ---------------------------------------------------------------------------
# Camera position relative to robot center
# ---------------------------------------------------------------------------

# Camera is approximately 14 cm in front of the robot center.
# Measure this value on the physical robot if greater precision is needed.
CAMERA_FORWARD_OFFSET = 0.14

# Map convention:
#   x = right
#   z = forward
CAMERA_OFFSET_FROM_ROBOT_CENTER = np.array([
    0.0,
    CAMERA_FORWARD_OFFSET
], dtype=float)


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
        landmark_radius=0.20,
        mirte_radius=0.22,
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

        to a landmark.
        """
        pos = np.asarray(pos, dtype=float)

        # Treat leaving the planned map area as collision.
        if np.any(pos < self.map_area[0]) or np.any(pos > self.map_area[1]):
            return 1

        clearance = self.landmark_radius + self.mirte_radius

        for landmark, landmark_id in self.landmarks:
            landmark = np.asarray(landmark, dtype=float)

            # Squared distance avoids an unnecessary square root.
            if np.sum((pos - landmark) ** 2) <= clearance ** 2:
                return 1

        return 0

    def get_map_from_mirte(self, mirte):
        """
        Take one camera frame and detect all unique ArUco landmarks.

        OpenCV's tvec points from the camera origin to the detected
        ArUco marker. Its units are the same as ARUCO_MARKER_LENGTH_MM,
        so the returned tvec values are in millimetres.

        We use tvec directly instead of applying an rvec-dependent
        box-center correction. The resulting 2D position is then shifted
        from the camera frame to the robot-center frame.
        """
        landmark_map = []

        img = mirte.get_image_compressed()

        if img is None:
            print("No camera image received.")
            return landmark_map

        corners, ids, _ = cv2.aruco.detectMarkers(
            img,
            arucoDict
        )

        if ids is None:
            print("No ArUco landmarks detected.")
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

            # IMPORTANT:
            # Use the raw tvec directly.
            #
            # tvec = [x, y, z] from camera origin to ArUco marker.
            # Because marker length is specified in mm, tvec is also in mm.
            x_mm, y_mm, z_mm = tvecs[i][0]

            # Convert camera-frame floor coordinates from mm to metres.
            landmark_camera = np.array([
                x_mm / 1000.0,
                z_mm / 1000.0
            ])

            # Camera -> robot-center transform.
            #
            # The camera is CAMERA_FORWARD_OFFSET metres in front of the
            # robot center, so the marker is that much farther from the
            # robot center along the forward z direction.
            landmark_pos = landmark_camera + self.camera_offset

            # Debug output. This is useful for checking whether two boxes
            # placed at the same physical depth get approximately equal z.
            print(
                f"ID {landmark_id:3d}: "
                f"raw camera x = {x_mm / 1000.0:+.3f} m, "
                f"raw camera z = {z_mm / 1000.0:+.3f} m, "
                f"robot x = {landmark_pos[0]:+.3f} m, "
                f"robot z = {landmark_pos[1]:+.3f} m"
            )

            landmark_map.append([
                landmark_pos,
                landmark_id
            ])

        return landmark_map

    def draw_map(self):
        """
        Basic plot used by RRT/debug code.
        """
        ax = plt.gca()

        clearance = self.landmark_radius + self.mirte_radius

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

        # Robot center.
        ax.scatter(
            0.0,
            0.0,
            marker="^"
        )

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
