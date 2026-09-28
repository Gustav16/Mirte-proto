# local_map.py
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
import time

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

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# IMPORTANT:
# Measure this on the physical robot.
#
# Your previous local_map.py used 0.10 m, although its comment said 8 cm.
# For the first test this is therefore kept at 0.10 m.
CAMERA_FORWARD_OFFSET = 0.14

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
OBJECT_DEPTH_M = 0.25
OBJECT_DEPTH_MM = OBJECT_DEPTH_M * 1000.0

# Shift from the ArUco paper plane toward the approximate box center.
obstacle_offset_3d_mm = np.array([
    0.0,
    0.0,
    OBJECT_DEPTH_MM / 2.0
])



#While we havent reached the target

target_id = None

def get_map_from_mirte(mirte):
        """
        Take one camera frame and detect all unique ArUco landmarks.

        Returned positions are expressed relative to the ROBOT CENTER.
        """
        landmark_map = []

        img = mirte.get_image_compressed()

        if img is None:
            return landmark_map

        corners, ids, _ = cv2.aruco.detectMarkers(img,arucoDict)

        if ids is None:
            return landmark_map

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, ARUCO_MARKER_LENGTH_MM, intrinsic_matrix, distortion_coeffs)

        seen_ids = set()

        for i in range(len(ids)):
            landmark_id = int(ids[i][0])

            if landmark_id in seen_ids:
                continue

            seen_ids.add(landmark_id)

            # Rotation of the marker relative to the camera.
            R, _ = cv2.Rodrigues(rvecs[i][0])

            # Estimate approximate center of the physical box.
            center_camera_mm = (tvecs[i][0] + R @ obstacle_offset_3d_mm)

            x_mm, y_mm, z_mm = center_camera_mm

            # Camera-frame 2D floor position in metres.
            landmark_camera = np.array([x_mm / 1000.0,z_mm / 1000.0])

            # Camera -> robot-center transform.
            # Camera is CAMERA_FORWARD_OFFSET in front of robot center,
            # therefore a landmark in front of the camera is that much
            # farther from the robot center.
            landmark_pos = (landmark_camera + CAMERA_OFFSET_FROM_ROBOT_CENTER)

            landmark_map.append([landmark_pos,landmark_id])

        return landmark_map

mirte = KU_Mirte()

while True:
    map, rvecs = get_map_from_mirte(mirte)
    print(map)

    print(rvecs)
    time.sleep(1)

# Finished successfully
del mirte