"""Drive to the goal with the RRT planner from Ex4 (Ex4/rrt/mirte_rrt.py), once MIRTE knows where it is.

The Ex4 planner works in MIRTE's own frame: MIRTE at (0, 0) facing +y, x to the right, in metres.
So the estimated pose (world frame, cm) is used to put the goal and the boxes into that frame: MIRTE's
position becomes (0, 0) and its heading is kept as "straight ahead". Then RRT plans around the boxes,
simplify_path shortens the path and Execute_path drives it (turn on the spot, then straight, leg by leg).
"""

import os
import sys

import numpy as np

# Ex4's planner. Appended (not inserted) so Ex5's own local_map stays the one that is used (same LocalMap class)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Ex4/rrt"))
from mirte_rrt import RRT, simplify_path, Execute_path
import robot_models
import local_map

PATH_RES = 0.1          # m, as in Ex4's main()
EXPAND_DIS = 0.4        # m, as in Ex4's main()
MAP_MARGIN = 0.5        # m, free space around start, goal and boxes for the RRT samples
PLAN_TRIES = 5          # RRT is random, try again if it does not find a path


def to_robot_frame(pose, point):
    """World point (cm) -> MIRTE's frame of the Ex4 planner (m): x to the right, y straight ahead."""
    x, y, theta = pose
    dx, dy = point[0] - x, point[1] - y
    ahead = dx * np.cos(theta) + dy * np.sin(theta)
    left = -dx * np.sin(theta) + dy * np.cos(theta)
    return np.array([-left, ahead]) / 100.0


def path_legs(path):
    """The (turn rad, distance cm) legs Execute_path will drive for a goal-first path, computed the same way."""
    points = list(reversed(path))
    legs, heading = [], 0.0
    for prev, point in zip(points[:-1], points[1:]):
        dx, dy = point[0] - prev[0], point[1] - prev[1]
        dist = np.hypot(dx, dy)
        if dist <= 1e-6:
            continue
        target = np.arctan2(-dx, dy)
        turn = np.arctan2(np.sin(target - heading), np.cos(target - heading))
        legs.append((turn, 100.0 * dist))
        heading = target
    return legs


def plan(pose, goal, box_centres):
    """Plan with Ex4's RRT from MIRTE (0, 0) to the goal. Returns the simplified path (goal first, like Ex4),
    or None. pose = (x, y, theta) and goal/box_centres in world cm."""
    goal_r = to_robot_frame(pose, goal)
    boxes_r = [to_robot_frame(pose, c) for c in box_centres]
    pts = np.array([[0.0, 0.0], goal_r] + boxes_r)
    # Same LocalMap as Ex4 (box radius 0.20 m + MIRTE radius 0.22 m), boxes as landmarks in MIRTE's frame
    m = local_map.LocalMap(landmarks=[[b, i] for i, b in enumerate(boxes_r)],
                           low=pts.min(axis=0) - MAP_MARGIN, high=pts.max(axis=0) + MAP_MARGIN)
    robot = robot_models.PointMassModel(ctrl_range=[-PATH_RES, PATH_RES])
    for _ in range(PLAN_TRIES):
        rrt = RRT(start=[0.0, 0.0], goal=goal_r, robot_model=robot, map=m,
                  expand_dis=EXPAND_DIS, path_resolution=PATH_RES)
        path = rrt.planning(animation=False)
        if path is not None:
            return simplify_path(path, rrt)
    return None


def drive(mirte, path):
    """Drive a planned path with Ex4's Execute_path. Returns the (turn rad, distance cm) legs it drove."""
    legs = path_legs(path)
    Execute_path(list(path), mirte) # Execute_path reverses the list it gets, so give it a copy
    return legs
