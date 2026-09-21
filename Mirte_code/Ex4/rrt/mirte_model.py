#Mirte Proto code
import numpy as np
from robot_models import RobotModel
import cv2 # Import the OpenCV library
import time

from pprint import *
import sys
import os




#change to fit directory
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '../../../Mirte/ku_mirte_python'
    )
)

from ku_mirte import KU_Mirte


"""
Some simple first-order robot dynamics models

This provides an extendable interface to make the simulation of robot motion independent of planning algorithms
The most basic RRT implementation just needs PointMassModel
"""

class MirteModel(RobotModel):

    def forward_dyn(self, x, u, T):
        path = []
        x_current = x.copy()

        for i in range(T):
            distance, dtheta = u[i]

            x_current[2] += dtheta
            x_current[0] += distance * np.cos(x_current[2])
            x_current[1] += distance * np.sin(x_current[2])

            # Only return x, y points
            path.append(x_current[:2].copy())

        return path

    def inverse_dyn(self, x, x_goal, T):
        dx = x_goal[0] - x[0]
        dy = x_goal[1] - x[1]

        distance = np.linalg.norm([dx, dy])

        if distance <= 1e-6:
            return [x[:2].copy()]

        # Direction towards goal
        target_theta = -np.arctan2(dy, dx)

        # Rotation needed from current orientation
        dtheta = target_theta - x[2]
        dtheta = np.arctan2(
            np.sin(dtheta),
            np.cos(dtheta)
        )

        u = []

        # Only rotate if necessary
        if abs(dtheta) > 1e-6:
            u.append(np.array([0.0, dtheta]))

        # Drive towards goal
        for _ in range(T - len(u)):
            step = min(self.ctrl_range[1], distance)

            u.append(np.array([step, 0.0]))

            distance -= step

            if distance <= 1e-6:
                break

        return self.forward_dyn(x, u, len(u))
