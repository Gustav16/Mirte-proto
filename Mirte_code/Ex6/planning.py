import local_map
from rrt import RRT, simplify_path
import robot_models
import numpy as np

#RRT init 
RRT_PATH_RES = 10      # cm
RRT_EXPAND_DIS = 40    # cm
RRT_MAP_MARGIN = 0.5    # m, free space around start, goal and boxes for the RRT samples
RRT_TRIES = 5

MAX_SEGMENT_LEN = 200 # cm, take a picture at least this often while driving

### planning methods ###

def rrt_plan(pos, m, goal, local_plan) -> list[list[float, float]] | None:

    start = np.array([pos[0], pos[1]])
    goal = np.array(goal)

    # Same LocalMap as Ex4 (box radius 0.20 m + MIRTE radius 0.22 m), the boxes as landmarks in the world frame
    robot = robot_models.PointMassModel(ctrl_range=[-RRT_PATH_RES, RRT_PATH_RES])

    size_array = [1, 1.25, 1.5, 1.75, 2]
    area = m.map_area.copy()

    for i in range(RRT_TRIES):  # RRT is random, try again if it does not find a path
        if local_plan:

            m.map_area = [
                np.asarray(bound, dtype=float) * size_array[i]
                for bound in area
            ]

        rrt = RRT(
            start=start,
            goal=goal,
            robot_model=robot,
            map=m,
            expand_dis=RRT_EXPAND_DIS,
            path_resolution=RRT_PATH_RES
        )

        path = rrt.planning(animation=False)

        if path is not None:
            result = split_long_segments(simplify_path(path, rrt), MAX_SEGMENT_LEN)

            if local_plan:
                m.map_area = area

            return result

    # Path could not be found, so reset map area if local plan
    if local_plan:
        m.map_area = area

    return None

def create_local_map(local_landmarks, local_obstacles, dist, previous_target, target) -> local_map.LocalMap: #TODO
    lm = local_map.LocalMap(low=(-100, 0.0),
    high=(100, dist+50), landmark_radius=20,
            mirte_radius=22, obstacale_radius=20)
    lmarks  =  []
    lobs = []
    for id in local_landmarks:
        pos = local_landmarks[id]
        radius = 20.0 if id in ([previous_target.id, target.id] if previous_target else [target.id]) else 40.0 + 20.0
        lmarks.append([pos, radius, id])
    for id in local_obstacles:
        pos = local_obstacles[id]
        lobs.append([pos, id])
    lm.landmarks = lmarks
    lm.obstacles = lobs
    return lm

def is_outisde_map(pos, radius, map_area) -> bool:
    "function cheking if an obstacle or landmark is outside a map area"
    #get difference between pos and closest point in the map
    delta = pos - np.clip(pos, map_area[0], map_area[1])
    return np.sum(delta**2) > radius**2


def should_replan(landmarks, obstales, new_landmarks, new_obstacles, map, curr_pose, target, global_plan) -> bool:
    "method for cheking if replan is need from camera observations"
    x, y, theta = curr_pose
    for id in new_landmarks:
        if id not in landmarks:
            if id == target.id:
                print(f'target found: {id}')
                return True
            #if we are locally planning when we can ignore all obstacales and landmarks outside the planning area
            
            if not global_plan:
                delta_x, delta_y = new_landmarks[id]
                pos = np.array([
                    x + delta_x * np.cos(theta) - delta_y * np.sin(theta),#TODO CHat says rotation are reversed
                    y + delta_x * np.sin(theta) + delta_y * np.cos(theta)])
                if is_outisde_map(pos, 60 ,map.map_area): #TODO set this properly
                    #landmark can safely be added as it is outise the map
                    landmarks[id] = pos
                    continue
            return True
    for id in new_obstacles:
            if id not in obstales:
                if not global_plan:
                    delta_x, delta_y = new_obstacles[id]
                    pos = np.array([
                        x + delta_x * np.cos(theta) - delta_y * np.sin(theta), #TODO CHat says rotation are reversed
                        y + delta_x * np.sin(theta) + delta_y * np.cos(theta)])
                    if is_outisde_map(pos, map.obstacale_radius, map.map_area):
                        #obstacle can safely be added as it is outise the map
                        obstales[id] = pos
                        continue
                return True
    return False


def split_long_segments(path, max_seg) -> list:
    """Split every segment longer than max_seg into equal parts. Path is goal first, as rrt_plan returns it."""
    points = [np.asarray(p, dtype=float) for p in reversed(path)] #start first
    new_path = [points[0]]
    for p in points[1:]:
        last = new_path[-1]
        n = int(np.ceil(np.linalg.norm(p - last) / max_seg))
        for k in range(1, n + 1):
            new_path.append(last + (p - last) * k / n)
    new_path.reverse()
    return new_path

