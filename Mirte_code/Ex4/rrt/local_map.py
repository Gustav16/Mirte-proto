#Mirte Proto code


import cv2 # Import the OpenCV library
import time

from pprint import *
import sys
import os
import math
import time
import numpy as np


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../../Mirte/ku_mirte_python'
    )
)

from ku_mirte import KU_Mirte

distortion_coeffs = np.zeros(5)

arucoMarkerLength = 145
f = 609.9
fx = f
fy = f

cx = 640 / 2
cy = 480 / 2

intrinsic_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float32)
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
Camera_offset = np.array([
            0,   # x: 0 cm left
            0.10   # z: 8 cm forward
        ])

object_depth = 0.20 #20 cm

obsticale_offset = np.array([
    0.0,
    0.0,
    object_depth / 2
])


class LocalMap:
    """
    Local map class
    """
    #landmark co
    def __init__(self, landmarks = [], landmark_radius=0.3, mirte_radius = 0.2, camera_offset = Camera_offset, low=(0, 0), high=(2, 2), res=0.1):
        self.landmarks = landmarks
        self.mirte_radius = mirte_radius
        self.landmark_radius = landmark_radius
        self.camera_offset = camera_offset
        self.map_area = [low, high]    #a rectangular area    

    def update(self, mirte):
        self.landmarks = self.get_map_from_mirte(mirte)

    def in_collision(self, pos):
        "collision query dunction"
        ###fast linear search as we dont expect many landmarks.
        #however data structures can be used for larger amounts of landmarks eg. kd-tres
        for landmark, landmark_id in self.landmarks:
            if (np.sum((pos - landmark)**2) <=( self.landmark_radius + self.mirte_radius)**2):
                return 1
        return 0

    def get_map_from_mirte(self, mirte):
        "function for gettig local map from mirte camera"
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
                    R, _ = cv2.Rodrigues(rvecs[i][0])
                    x, y, z = (tvecs[i][0] + R @ obsticale_offset)/ 1000.0 #get center of landmark and convert to meter
                    landmark = np.array([x, z])
                    # Correct camera position → robot-center position
                    landmark -= self.camera_offset

                    landmark_map.append([landmark, landmark_id])
        return landmark_map
    def draw_map(self):
        #note the x-y axes difference between imshow and plot
        pass
        
        



