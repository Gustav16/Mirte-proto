"""

Path planning with Randomized Rapidly-Exploring Random Trees (RRT)

Adapted from 
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.animation import FFMpegWriter
from robot_models import RobotModel

import cv2 # Import the OpenCV library
import time

from pprint import *
import sys
import os
from ku_mirte import KU_Mirte
import grid_occ, robot_models, local_map


MAX_DRIVE_DIST = 2.0

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
    os.path.join(
        os.path.dirname(__file__),
        '../../../Mirte/ku_mirte_python'
    )
)


def simplify_path(path, rrt):
    # rrt.planning() returnerer ruten fra mål til start, så den vendes,
    # og vi arbejder fra start mod mål
    points = [np.asarray(p, dtype=float) for p in reversed(path)]

    simplified = [points[0]]      # startpunktet er altid med
    anchor = 0                    # indeks for det punkt, vi står i nu
    last = len(points) - 1        # indeks for målet

    while anchor < last:
        from_node = RRT.Node(points[anchor])

        # Søg bagfra: find det FJERNESTE punkt, der kan nås i en lige linje
        for i in range(last, anchor, -1):
            new_node = rrt.steer(from_node, RRT.Node(points[i]))  # ingen længdegrænse
            if rrt.check_collision_free(new_node):
                simplified.append(points[i])
                anchor = i
                break
        else:
            # Burde ikke ske (nabopunktet er altid frit), men undgår en uendelig løkke
            anchor += 1
            simplified.append(points[anchor])

    # Tilbage til samme format som før (mål først), så resten af koden virker uændret
    simplified.reverse()
    return simplified