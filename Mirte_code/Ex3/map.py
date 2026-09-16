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



distortion_coeffs = np.zeros(5)

#set values
arucoMarkerLength = 145
intrinsic_matrix = None
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


#While we havent reached the target

radius = None

#change to while not presed q waitkey
while cv2.waitKey(1) != ord('q'):
    landmark_map = []
    img = mirte.get_image_compressed()
    corners, ids, _ = cv2.aruco.detectMarkers(img, arucoDict)


    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        arucoMarkerLength,
        intrinsic_matrix,
        distortion_coeffs
    )
    id_set = set()
    for i in range(len(ids)):
        landmark_id = ids[i][0]
        if landmark_id not in id_set:
            id_set.add(landmark_id)
            x, y, z = tvecs[i][0]
            landmark_map.append([(x, z), landmark_id])
    print(landmark_map)
    time.sleep(0.3)

# Finished successfully
del mirte