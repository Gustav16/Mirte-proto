import local_map
from rrt import RRT, simplify_path
import robot_models
import numpy as np

#RRT init 
RRT_PATH_RES = 10      # cm
RRT_EXPAND_DIS = 40    # cm
RRT_MAP_MARGIN = 0.5    # m, free space around start, goal and boxes for the RRT samples
RRT_TRIES = 5

RRT_TRIES_GLOBAL = 2       # global retries search the same map, so keep few
RRT_ITER_LOCAL = 500
RRT_ITER_GLOBAL = 300      # failed global searches stop sooner
GLOBAL_GOAL_RATE = 15      # % of samples that are the goal (local keeps 5)

MAX_SEGMENT_LEN = 200 # cm, take a picture at least this often while driving

### planning methods ###

def rrt_plan(pos, m, goal, local_plan, tries=None) -> list[list[float, float]] | None:

    start = np.array([pos[0], pos[1]], dtype=float)
    goal = np.array(goal, dtype=float)

    # a blocked goal or start can never succeed, so don't spend any RRT time on it
    if m.in_collision(goal):
        return None
    if m.in_collision(start):
        print("rrt_plan: start is inside a clearance zone, skipping")
        return None

    if tries is None:
        tries = RRT_TRIES if local_plan else RRT_TRIES_GLOBAL

    robot = robot_models.PointMassModel(ctrl_range=[-RRT_PATH_RES, RRT_PATH_RES])

    size_array = [1, 1.25, 1.5, 1.75, 2]
    area = m.map_area.copy()

    try:
        for i in range(min(tries, len(size_array))):
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
                path_resolution=RRT_PATH_RES,
                goal_sample_rate=5 if local_plan else GLOBAL_GOAL_RATE,
                max_iter=RRT_ITER_LOCAL if local_plan else RRT_ITER_GLOBAL
            )

            path = rrt.planning(animation=False)

            if path is not None:
                return split_long_segments(simplify_path(path, rrt), MAX_SEGMENT_LEN)
    finally:
        m.map_area = area   # always restore, whichever way we leave

    return None


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

