#Mirte Proto code
import numpy as np
from robot_models import RobotModel


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

import numpy as np
from robot_models import RobotModel


class MirteModel(RobotModel):

    def forward_dyn(self, x, u, T):
        path = [x]

        for i in range(T):
            distance, dtheta = u[i]

            x_new = path[-1].copy()

            x_new[2] += dtheta
            x_new[0] += distance * np.cos(x_new[2])
            x_new[1] += distance * np.sin(x_new[2])

            path.append(x_new)
            
        return path[1:]

    def inverse_dyn(self, x, x_goal, T):
        x_current = x.copy()
        u = []

        dx = x_goal[0] - x_current[0]
        dy = x_goal[1] - x_current[1]

        dist = np.linalg.norm([dx, dy])

        if dist == 0:
            return []

        target_theta = -np.arctan2(dy, dx)

        dtheta = target_theta - x_current[2]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        # Only turn if necessary
        if abs(dtheta) > 1e-6:
            u.append(np.array([0.0, dtheta]))
            x_current[2] += dtheta

        # Use up to T steps to drive towards the goal
        for i in range(T - len(u)):
            dx = x_goal[0] - x_current[0]
            dy = x_goal[1] - x_current[1]

            dist = np.linalg.norm([dx, dy])

            if dist <= 1e-6:
                break

            step = min(self.ctrl_range[1], dist)

            u.append(np.array([step, 0.0]))

            x_current[0] += step * np.cos(x_current[2])
            x_current[1] += step * np.sin(x_current[2])

        return self.forward_dyn(x, u, len(u))

