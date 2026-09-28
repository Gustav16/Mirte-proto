"""
Some simple first-order robot dynamics models

This provides an extendable interface to make the simulation of robot motion independent of planning algorithms
The most basic RRT implementation just needs PointMassModel
"""
import numpy as np

class RobotModel:

    def __init__(self, ctrl_range) -> None:
        #record the range of action that can be used to integrate the state
        self.ctrl_range = ctrl_range
        return
    
    def forward_dyn(self, x, u, T):
        #need to return intergated path of X with u along the horizon of T
        return NotImplementedError

    def inverse_dyn(self, x, x_goal, T):
        #return dynamically feasible path to move to the x_goal as close as possible
        return NotImplementedError

class PointMassModel(RobotModel):
    #Note Arlo is differential driven and may be simpler to avoid Dubins car model by rotating in-place to direct and executing piecewise straight path  
    #this is the "noise-free" motion model: shift x according to a command of u (velocity*dt).
    #it supports u of T steps and return the full trajectory. Just use T=1 if only one step is considered.
    def forward_dyn(self, x, u, T):
        path = [x]
        #note u must have T ctrl to apply
        for i in range(T):
            x_new = path[-1] + u[i] #u is velocity command here
            path.append(x_new)    
        
        return path[1:]

    def inverse_dyn(self, x, x_goal, T):
        #this is the inverse of motion model: what u of T steps can realize x_goal or move as close as possible given the current state x
        #return realized trajectory. this would be used to check if the trajectory would be incollision or not.
        #for point mass, the path is just a straight line by taking full ctrl_range at each step
        dist = np.linalg.norm(x_goal-x)
        dir = (x_goal-x)/dist
        #we cannot move more than the maximum control range at each step; clip it to make move as close as possible
        u = np.array([dir*min( [self.ctrl_range[1], dist/float(T)] ) for _ in range(T)])

        return self.forward_dyn(x, u, T)

class MirteModel(RobotModel):

    def __init__(self, ctrl_range):
        super().__init__(ctrl_range)
        self.theta = 0.0

    def forward_dyn(self, x, u, T):
        path = []

        x_current = np.array(x, dtype=float)

        for i in range(T):
            distance, dtheta = u[i]

            self.theta += dtheta

            x_current[0] += distance * np.cos(self.theta)
            x_current[1] += distance * np.sin(self.theta)

            path.append(x_current.copy())

        return path

    def inverse_dyn(self, x, x_goal, T):
        dx = x_goal[0] - x[0]
        dy = x_goal[1] - x[1]

        distance = np.linalg.norm([dx, dy])

        if distance <= 1e-6:
            return [x.copy()]

        target_theta = np.arctan2(dy, dx)

        dtheta = target_theta - self.theta
        dtheta = np.arctan2(
            np.sin(dtheta),
            np.cos(dtheta)
        )

        u = []

        # Rotate only if necessary
        if abs(dtheta) > 1e-6:
            u.append(np.array([0.0, dtheta]))

        # Drive towards target
        for _ in range(T - len(u)):
            step = min(self.ctrl_range[1], distance)

            u.append(np.array([step, 0.0]))

            distance -= step

            if distance <= 1e-6:
                break

        return self.forward_dyn(x, u, len(u))
