# Shared RRT, mapping and driving functions for MIRTE.

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../Mirte/ku_mirte_python"
    )
)

import local_map
import robot_models

from mcl import MCL
from path_follower import follow_path
from path_smoothing import smooth_path
from visualize_local_map import plot_local_map, plot_path


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
            if i > 0 and i % 100 == 0:
                print(
                    f"RRT iteration {i}/{self.max_iter}"
                )

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


def make_map(mirte):
    """
    Make a local map from two camera frames.

    An ArUco ID must be visible in both frames. This removes many
    one-frame false detections, such as an unexpected ID 120.
    Positions from the two frames are averaged.
    """
    world_map = local_map.LocalMap(
        low=(-1.5, 0.0),
        high=(1.5, 3.0)
    )

    first = world_map.get_map_from_mirte(mirte)
    second = world_map.get_map_from_mirte(mirte)

    first_dict = {
        int(marker_id): np.asarray(position, dtype=float)
        for position, marker_id in first
    }
    second_dict = {
        int(marker_id): np.asarray(position, dtype=float)
        for position, marker_id in second
    }

    confirmed = []

    for marker_id in first_dict.keys() & second_dict.keys():
        position = (
            first_dict[marker_id]
            + second_dict[marker_id]
        ) / 2.0

        confirmed.append([
            position,
            marker_id
        ])

    world_map.landmarks = confirmed
    return world_map


def get_aruco_observations(mirte, sensor_map):
    """
    Take a fresh camera measurement.

    Returns:
        [(marker_id, distance, bearing), ...]
    """
    detected = sensor_map.get_map_from_mirte(mirte)
    observations = []

    for position, marker_id in detected:
        x, z = np.asarray(position, dtype=float)

        observations.append(
            (
                int(marker_id),
                float(math.hypot(x, z)),
                float(math.atan2(-x, z)),
            )
        )

    return observations



def direct_path_is_free(world_map, start, goal, step=0.05):
    """Check a straight line from start to goal."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    distance = np.linalg.norm(goal - start)

    if distance < 1e-9:
        return True

    number_of_points = max(
        2,
        int(math.ceil(distance / step)) + 1
    )

    for t in np.linspace(0.0, 1.0, number_of_points):
        point = start + t * (goal - start)

        if world_map.in_collision(point):
            return False

    return True


def make_direct_path(start, goal, spacing=0.05):
    """Create evenly spaced points on a straight path."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    distance = np.linalg.norm(goal - start)

    if distance < 1e-9:
        return [start, goal]

    number_of_segments = max(
        1,
        int(math.ceil(distance / spacing))
    )

    return [
        start + t * (goal - start)
        for t in np.linspace(
            0.0,
            1.0,
            number_of_segments + 1
        )
    ]

def plan_path(world_map, goal):
    """
    Plan a local path from [0, 0] to goal.

    First try the straight path. RRT is only used when an obstacle
    actually blocks the direct route.
    """
    start = np.array([0.0, 0.0])
    goal = np.asarray(goal, dtype=float)

    if np.linalg.norm(goal) < 0.03:
        return [start, goal]

    print("Planning path...")

    if direct_path_is_free(
        world_map,
        start,
        goal
    ):
        print("Direct path is free.")
        return make_direct_path(
            start,
            goal,
            spacing=0.05
        )

    print("Direct path blocked. Starting RRT...")

    path_resolution = 0.10

    robot = robot_models.PointMassModel(
        ctrl_range=[
            -path_resolution,
            path_resolution
        ]
    )

    rrt = RRT(
        start=start,
        goal=goal,
        robot_model=robot,
        map=world_map,
        expand_dis=0.40,
        path_resolution=path_resolution,
        max_iter=500,
    )

    path = rrt.planning(
        animation=False,
        writer=None
    )

    if path is None:
        print("RRT could not find a path.")
        return None

    print("RRT path found.")

    # RRT returns goal -> start.
    path = list(reversed(path))

    return smooth_path(
        path,
        world_map,
        spacing=0.05
    )


def drive_path(mirte, world_map, path):
    """
    Follow one local path.

    A fresh MCL instance is used for this local movement.
    After the movement, exploration takes a new camera map and can plan again.
    """
    localizer = MCL(
        world_map,
        number_of_particles=300
    )

    fixed_landmarks = list(world_map.landmarks)

    def observations():
        return get_aruco_observations(
            mirte,
            world_map
        )

    follow_path(
        path,
        mirte,
        localizer=localizer,
        get_observations=observations,
        landmarks=fixed_landmarks,
    )


def save_path_plot(world_map, path, filename):
    """Save the most recently planned local path."""
    plot_local_map(world_map.landmarks)
    plot_path(
        path,
        start=path[0],
        goal=path[-1]
    )

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved plot to: {filename}")
