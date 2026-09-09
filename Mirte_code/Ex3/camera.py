import cv2 # Import the OpenCV library
import time

from pprint import *
import sys
import os
import math
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Mirte/ku_mirte_python'))
from Mirte.ku_mirte_python.ku_mirte import KU_Mirte

time.sleep(1)  # wait for camera to setup


#program start

mirte = KU_Mirte()

LIN_speed = 0.35

#set success dist
success_distance = None
reached_target = False

distortion_coeffs = np.zeros(5)

#set values
arucoMarkerLength = None
intrinsic_matrix = None
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


#While we havent reached the target

target_id = None


while not reached_target:

    img = mirte.get_image_compressed()
    corners, ids, _ = cv2.aruco.detectMarkers(img, arucoDict)

    # ----Search----
    if target_id is None:

        if ids is None or ids.size == 0:
            # Keep rotating until we see an ArUco
            mirte.drive(0, 0.2, 1, blocking=False)
            time.sleep(0.2)
            continue

        # We found one -> lock onto it
        target_id = ids[0, 0]
        print("Acquired target:", target_id)

    # ----Tracking----
    indices = np.where(ids.flatten() == target_id)[0] \
        if ids is not None else []

    if len(indices) == 0:
        # Lost target -> go back to SEARCHING
        target_id = None
        mirte.stop()
        continue

    i = indices[0]

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        arucoMarkerLength,
        intrinsic_matrix,
        distortion_coeffs
    )

    target_tvec = tvecs[i][0]

    #have we reached target
    if np.linalg.norm(target_tvec) < success_distance:
        mirte.stop()
        reached_target = True
        break

    x, y, z = target_tvec
    angle = np.arctan2(x, z)

    # Drive toward target
    mirte.drive(LIN_speed, angle, 4, blocking=False)

    time.sleep(0.3)



# Finished successfully
del mirte