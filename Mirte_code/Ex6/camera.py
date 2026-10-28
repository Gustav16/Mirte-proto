import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../Mirte/ku_mirte_python"
    )
)

#from ku_mirte import KU_Mirte


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
CAMERA_FORWARD_OFFSET = 14

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
OBJECT_DEPTH_M = 0.22
OBJECT_DEPTH_MM = OBJECT_DEPTH_M * 1000.0

# Shift from the ArUco paper plane toward the approximate box center.
obstacle_offset_3d_MM = np.array([
    0.0,
    0.0,
    -OBJECT_DEPTH_MM / 2.0
])

def get_observations(landmarks, mirte):
    "Method for getting th mirtes local observations"
    z = {}
    local_landmarks = {}
    local_obstacles = {}

    #time.sleep(3)
    img = mirte.get_next_image()
    
    if img is None:
        raise ValueError('could not retrive image')
        #return z, local_landmarks, local_obstacles
    
    corners, ids, _ = cv2.aruco.detectMarkers(img,arucoDict)

    if ids is None:
        return z, local_landmarks, local_obstacles

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, ARUCO_MARKER_LENGTH_MM, intrinsic_matrix, distortion_coeffs)

    seen_ids = set()

    for i in range(len(ids)):
        id = int(ids[i][0])

        if id in seen_ids:
            continue

        seen_ids.add(id)

        # Rotation of the marker relative to the camera.
        R, _ = cv2.Rodrigues(rvecs[i][0])

        # Estimate approximate center of the physical box.
        center_camera_mm = (tvecs[i][0] + R @ obstacle_offset_3d_MM)

        x_mm, y_mm, z_mm = center_camera_mm

        # Camera-frame 2D floor position in centimetres.
        pos = np.array([x_mm / 10.0,z_mm / 10.0])

        # Camera -> robot-center transform.
        # Camera is CAMERA_FORWARD_OFFSET in front of robot center,
        # therefore a landmark in front of the camera is that much
        # farther from the robot center.
        pos = (pos + CAMERA_OFFSET_FROM_ROBOT_CENTER)

        if id in landmarks:
            local_landmarks[id] = pos
        else:
            local_obstacles[id] = pos
    for id in local_landmarks:
        pos = local_landmarks[id]
        #TODO check if this is correct
        z[id] = [np.linalg.norm(pos), np.arctan2(-pos[0], pos[1])]
    return z, local_landmarks, local_obstacles

def print_observations(z, local_landmarks, local_obstacles) -> None:
    "method used for printing observations"
    if z:
        print("Landmarks seen:")
    for id in z:
        print(f"Landmark id = {id}, dist = {z[id][0]}, angle = {z[id][1]} local pos: x = {local_landmarks[id][0]}, y = {local_landmarks[id][1]}")
    if local_obstacles:
        print("Obstacles seen:")
    for id in local_obstacles:
        print(f"Obstacle id = {id} local pos: x = {local_obstacles[id][0]}, y = {local_obstacles[id][1]}")