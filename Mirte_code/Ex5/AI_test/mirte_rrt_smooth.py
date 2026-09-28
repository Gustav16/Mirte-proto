# Mirte Proto code

import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../Mirte/ku_mirte_python"
    )
)

from ku_mirte import KU_Mirte

import local_map
import robot_models

from between import find_between_goal
from mcl import MCL
from path_follower import follow_path
from path_smoothing import smooth_path
from visualize_local_map import plot_local_map, plot_path, PNG_PATH


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, pos):
            self.pos = np.asarray(pos, dtype=float)      #configuration position, usually 2D/3D for planar robots  
            self.path = []      #the path with a integration horizon. this could be just a straight line for holonomic system
            self.parent = None
        
        def calc_distance_to(self, to_node):
            # node distance can be nontrivial as some form of cost-to-go function for e.g. underactuated system
            # use euclidean norm for basic holonomic point mass or as heuristics
            d = np.linalg.norm(np.array(to_node.pos) - np.array(self.pos))
            return d
        
    def __init__(self,
                 start,
                 goal,
                 robot_model,   #model of the robot
                 map,           #map should allow the algorithm to query if the path in a node is in collision. note this will ignore the robot geom
                 expand_dis=0.2,
                 path_resolution=0.05,
                 goal_sample_rate=5,
                 max_iter=500,
                 ):

        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.robot = robot_model
        self.map = map
        
        self.min_rand = map.map_area[0]
        self.max_rand = map.map_area[1]

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter

        self.node_list = []

    def planning(self, animation=True, writer=None):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision_free(new_node):
                self.node_list.append(new_node)

            #try to steer towards the goal if we are already close enough
            if self.node_list[-1].calc_distance_to(self.end) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision_free(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation:
                self.draw_graph(rnd_node)
                if writer is not None:
                    writer.grab_frame()

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        # integrate the robot dynamics towards the sampled position
        # for holonomic point pass robot, this could be straight forward as a straight line in Euclidean space
        # while need some local optimization to find the dynamically closest path otherwise
        new_node = self.Node(from_node.pos)
        d = new_node.calc_distance_to(to_node)

        new_node.path = [new_node.pos]

        if extend_length > d:
            extend_length = d

        n_expand = int(extend_length // self.path_resolution)

        if n_expand > 0:
            steer_path = self.robot.inverse_dyn(new_node.pos, to_node.pos, n_expand)
            #use the end position to represent the current node and update the path
            new_node.pos = steer_path[-1]
            new_node.path += steer_path

        d = new_node.calc_distance_to(to_node)
        if d <= self.path_resolution:
            #this is considered as connectable
            new_node.path.append(to_node.pos)

            #so this position becomes the representation of this node
            new_node.pos = to_node.pos.copy()

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [self.end.pos]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.pos)
            node = node.parent
        path.append(node.pos)

        return path

    def get_random_node(self):
        if np.random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                np.random.uniform(self.map.map_area[0], self.map.map_area[1])
                )
        else:  # goal point sampling
            rnd = self.Node(self.end.pos)
        return rnd

    def draw_graph(self, rnd=None):
        # plt.clf()
        # # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect(
        #     'key_release_event',
        #     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.pos[0], rnd.pos[1], "^k")

        # draw the map
        self.map.draw_map()

        for node in self.node_list:
            if node.parent:
                path = np.array(node.path)
                plt.plot(path[:, 0], path[:, 1], "-g")

        plt.plot(self.start.pos[0], self.start.pos[1], "xr")
        plt.plot(self.end.pos[0], self.end.pos[1], "xr")
        plt.axis(self.map.extent)
        plt.grid(True)
        plt.pause(0.01)


    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [ node.calc_distance_to(rnd_node)
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind


    def check_collision_free(self, node):
        if node is None:
            return False
        for p in node.path:
            if self.map.in_collision(np.array(p)):
                return False
        return True


sys.path.append(
    os.path.join(os.path.dirname(__file__), '../../../Mirte/ku_mirte_python')
)

from ku_mirte import KU_Mirte
import robot_models, local_map
from between import find_between_goal
from visualize_local_map import plot_local_map, plot_path, PNG_PATH
from path_smoothing import smooth_path
from path_follower import follow_path
from mcl import MCL


# --- Runtime settings -------------------------------------------------------
PATH_RESOLUTION = 0.10
EXPAND_DISTANCE = 0.40
RRT_MAX_ITER = 1500
MCL_PARTICLES = 1000


def observations_from_local_map(sensor_map):
    """Convert robot-relative ArUco [x,z] positions to MCL range+bearing."""
    observations = []
    for position, marker_id in sensor_map.landmarks:
        x, z = np.asarray(position, dtype=float)
        observations.append({
            "id": int(marker_id),
            "range": float(np.hypot(x, z)),
            "bearing": float(np.arctan2(x, z)),
        })
    return observations


def landmark_world_map(planning_map):
    """Fixed landmark map in the coordinate frame at program start."""
    return {
        int(marker_id): np.asarray(position, dtype=float).copy()
        for position, marker_id in planning_map.landmarks
    }


def make_localizer(planning_map):
    """Initial belief around MIRTE's known start pose [0,0,0]."""
    prior = np.empty((MCL_PARTICLES, 3), dtype=float)
    prior[:, 0] = np.random.normal(0.0, 0.03, MCL_PARTICLES)
    prior[:, 1] = np.random.normal(0.0, 0.03, MCL_PARTICLES)
    prior[:, 2] = np.random.normal(0.0, np.deg2rad(4.0), MCL_PARTICLES)
    return MCL(prior)


def get_aruco_observations(mirte, localizer_map):
    """
    Return observations as:
        (marker_id, distance, bearing)

    local_map.get_landmarks() already converts detections to positions
    relative to the robot centre.
    """
    detected = localizer_map.get_landmarks(mirte)
    observations = []

    for position, marker_id in detected:
        x, z = position
        distance = math.hypot(x, z)
        bearing = math.atan2(-x, z)

        observations.append(
            (int(marker_id), distance, bearing)
        )

    return observations


def main():
    path_resolution = 0.10

    mirte = KU_Mirte()
    time.sleep(1)

    # --------------------------------------------------
    # 1. Build local map
    # --------------------------------------------------
    world_map = local_map.LocalMap(
        low=(-1, 0),
        high=(1, 2)
    )
    world_map.update(mirte)

    print("landmark amount:", len(world_map.landmarks))

    # --------------------------------------------------
    # 2. Find a goal between landmarks
    # --------------------------------------------------
    goal, gate_ids = find_between_goal(world_map)

    if goal is None:
        print("Could not find a goal between landmarks.")
        del mirte
        return

    print(
        f"using passage between IDs "
        f"{gate_ids[0]} and {gate_ids[1]}"
    )
    print(
        f"goal: x={goal[0]:+.3f}, "
        f"z={goal[1]:+.3f}"
    )

    # --------------------------------------------------
    # 3. Plan path with the existing RRT
    # --------------------------------------------------
    robot = robot_models.PointMassModel(
        ctrl_range=[-path_resolution, path_resolution]
    )

    rrt = RRT(
        start=[0, 0],
        goal=goal,
        robot_model=robot,
        map=world_map,
        expand_dis=0.4,
        path_resolution=path_resolution,
    )

    path = rrt.planning(
        animation=False,
        writer=None
    )

    if path is None:
        print("Cannot find path.")
        del mirte
        return

    print("found path!!")

    # RRT returns goal -> start. The follower uses start -> goal.
    path = list(reversed(path))

    # --------------------------------------------------
    # 4. Simplify the path
    # --------------------------------------------------
    smooth = smooth_path(
        path,
        world_map,
        spacing=0.05
    )

    print("raw path points:", len(path))
    print("smooth path points:", len(smooth))

    # --------------------------------------------------
    # 5. Create MCL localizer
    # --------------------------------------------------
    localizer = MCL(
        world_map,
        number_of_particles=300
    )

    # Landmarks stay fixed in the coordinate system in which
    # the initial local map was created.
    fixed_landmarks = list(world_map.landmarks)

    def observations():
        return get_aruco_observations(
            mirte,
            world_map
        )

    print("initial pose:", localizer.estimate_pose())

    # --------------------------------------------------
    # 6. Follow the smoothed path
    # --------------------------------------------------
    follow_path(
        smooth,
        mirte,
        localizer=localizer,
        get_observations=observations,
        landmarks=fixed_landmarks,
    )

    # --------------------------------------------------
    # 7. Save plot
    # --------------------------------------------------
    plot_local_map(world_map.landmarks)
    plot_path(
        smooth,
        start=smooth[0],
        goal=smooth[-1]
    )

    plt.savefig(
        PNG_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"Saved plot to: {PNG_PATH}")

    del mirte


if __name__ == "__main__":
    main()
