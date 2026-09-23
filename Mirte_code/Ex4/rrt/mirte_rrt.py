"""

Path planning with Randomized Rapidly-Exploring Random Trees (RRT)

Adapted from 
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.animation import FFMpegWriter
import numpy as np
from robot_models import RobotModel

import cv2 # Import the OpenCV library
import time

from pprint import *
import sys
import os

class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, pos):
            self.pos = pos      #configuration position, usually 2D/3D for planar robots  
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

from ku_mirte import KU_Mirte
import grid_occ, robot_models, local_map
from visualize_local_map import plot_local_map, plot_path, PNG_PATH


def Execute_path(path, mirte):
    path.reverse()
    current_pose = np.array([0.0, 0.0, 0.0])

    linear_speed = 0.3   # m/s
    angular_speed = 0.5   # rad/s

    for point in path[::-1][1:]:
        dx = point[0] - current_pose[0]
        dy = point[1] - current_pose[1]

        distance = np.linalg.norm([dx, dy])

        if distance <= 1e-6:
            continue

        # Direction towards next point
        target_theta = -np.arctan2(dy, dx)

        # Required rotation
        dtheta = target_theta - current_pose[2]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        # Rotate at fixed angular speed
        if abs(dtheta) > 1e-6:
            mirte.drive(
                0.0,
                np.sign(dtheta) * angular_speed,
                abs(dtheta) / angular_speed
            )

        # Drive at fixed linear speed
        mirte.drive(
            linear_speed,
            0.0,
            distance / linear_speed
        )

        # Update estimated pose
        current_pose[0] = point[0]
        current_pose[1] = point[1]
        current_pose[2] = target_theta


def main():
    path_res = 0.1 #10 cm
    mirte = KU_Mirte()
    time.sleep(1)  # wait for camera to setup
    map = local_map.LocalMap()
    map.update(mirte)
    robot = robot_models.PointMassModel(ctrl_range=[-path_res, path_res])

    #standard goal destination
    goal = [0, 1.9]

    #go betweem 2 landmarks
    # landmark_count = len(map.landmarks)
    # if landmark_count > 1:
    #     for i in range(landmark_count-1):
            
    #         rrt = RRT(
    #                 start=[0, 0],
    #                 goal=[0, 1.9],
    #                 robot_model=robot,
    #                 map=map,
    #                 expand_dis=0.4, #0.4 meters
    #                 path_resolution=path_res, #10 cm
    #                 )
    #         show_animation = False
    #         writer = None
    
    rrt = RRT(
        start=[0, 0],
        goal=[0, 1.9],
        robot_model=robot,
        map=map,
        expand_dis=0.4, #0.4 meters
        path_resolution=path_res, #10 cm
        )
    
    show_animation = False
    metadata = dict(title="RRT Test")
    #writer = FFMpegWriter(fps=15, metadata=metadata)
    writer = None
    fig = plt.figure()
    if writer is not None:
        with writer.saving(fig, "rrt_test.mp4", 100):
            path = rrt.planning(animation=show_animation, writer=writer)

            if path is None:
                print("Cannot find path")
            else:
                print("found path!!")

                # Draw final path
                if show_animation:
                    rrt.draw_graph()
                    plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                    plt.grid(True)
                    plt.pause(0.01)  # Need for Mac
                    plt.show()
                    writer.grab_frame()
    else:
        #do not save videos
        path = rrt.planning(animation=show_animation, writer=writer)

        if path is None:
            print("Cannot find path")
        else:
            print("found path!!")
            print(path)

            # Draw local map + planned route in the same graph
            # (generate_final_course returns the path goal-first)
            plot_local_map(map.landmarks)
            plot_path(path, start=path[-1], goal=path[0])
            plt.savefig(PNG_PATH, dpi=200, bbox_inches="tight")
            print(f"Saved plot to: {PNG_PATH}")
            plt.pause(0.01)  # Need for Mac

            #execute path
            Execute_path(path, mirte)
    del mirte

if __name__ == '__main__':
    main()