#import packages
import local_map
from mcl import MCL
import particle as pcl
import camera as cam
from rrt import RRT, simplify_path
import robot_models
import numpy as np
import time
import drive


from ku_mirte import KU_Mirte
#init landmark
landmarkIDs = [1,2,3,4]


#init landmarks
landmarks = {
    landmarkIDs[0]: (0.0, 0.0),
    landmarkIDs[1]: (0.0, 300.0),
    landmarkIDs[2]: (400.0, 0.0),  
    landmarkIDs[3]: (300.0, 400.0)  
}

### Target class ###
class Target:
    "class for targets so we can pass them as pointers"
    def __init__(self, id):
        self.id = id
    def create_target_stack(target_stack):
        return [Target(id) for id in target_stack]

#init target
 
target = None
previous_target = None


#init target stack

target_stack = Target.create_target_stack([landmarkIDs[0],landmarkIDs[3],landmarkIDs[2],landmarkIDs[1], landmarkIDs[0]])

#init map
LANDMARK_RADIUS = 20
global_map = local_map.LocalMap(low=(-120.0, -120.0),
        high=(420.0, 520.0), landmark_radius=LANDMARK_RADIUS,
                mirte_radius=22) #TODO update local map so it can set radius for different landmarks

Local_MAP = None

global_map.landmarks = [[list(landmarks[ID]), 20.0 + 40.0, ID] for ID in landmarkIDs] # same landmarks as above, in the map's format
#update local map for our local map that acccounts for obstacles

#init needed objects and constants

VALID_RANGE = 40
SUCCESS__RANGE = LANDMARK_RADIUS + VALID_RANGE

#number of particles 
NUM_PARTICLES = 1000

#init particles
particles = pcl.initialize_particles(NUM_PARTICLES,global_map) #init prior

#init mcl r
aug_mcl = MCL(particles)

#init mirte
mirte = KU_Mirte()
#sleep to let camera start
time.sleep(0.4)

#convergence constants
X_VAR_ERR = 30**2
Y_VAR_ERR = 30**2
THETA_VAR_ERR = 20**2


#RRT init 
RRT_PATH_RES = 10      # cm
RRT_EXPAND_DIS = 40    # cm
RRT_MAP_MARGIN = 0.5    # m, free space around start, goal and boxes for the RRT samples
RRT_TRIES = 5

#Turn variables
HAS_TURNED_LEFT = False
HAS_TURNED_RIGHT = False
turns = 0

#max turn constant
SCAN_STEP = np.radians(20)
MAX_TURNS = (np.radians(360)//SCAN_STEP) - 2

### Method for reseting turning varibles ###
def reset_turn_variables() -> None:
    "method for reseting varibles used in turning scheme"
    global HAS_TURNED_LEFT, HAS_TURNED_RIGHT, turns

    HAS_TURNED_LEFT = False
    HAS_TURNED_RIGHT = False
    turns = 0

### Methods for setting and updating target ###
def pop(target_stack, m) -> Target:
    "returns the new target"
    if target_stack:
        new_target = target_stack.pop()
        #downsize border restriction
        m.update_milestone_radius(new_target.id, 20)
        return new_target
    raise ValueError('stack empty but pop was called')
        
def target_update(target, previous_target, m) -> tuple[None, Target]:
    "method for updating target"
    if previous_target:
        #update and increase border restriction
        m.update_milestone_radius(previous_target.id, 60)
    return None, target

### Target success method ###
def has_reached_target(target, z, est_pose, has_converged) -> bool: #TODO should we do so it get the map instead ?
    "method cheking if tagrget has been reached"
    id = target.id
    #if we can see can we determine sucess
    if id in z and z[id][0] <= SUCCESS__RANGE:
        return True
    #if we cant see but have converged poition, are we in the success range
    #TODO maybe check est_pose isnt inside the target
    if has_converged and np.linalg.norm([est_pose.getX(), est_pose.getY()], [landmarks[id][0], landmarks[id][1]]) <= SUCCESS__RANGE:
        return True
    #TODO check if there are other success conditions
    return False

### planning methods ###

def rrt_plan(pos, m, goal) -> list[list[float, float]]| None: #TODO
    """Plan in the world frame (m) from MIRTE's estimated position (from the measurements and motion updates of the
    particle filter) to GOAL around the boxes with mirte_rrt's RRT, then shorten it with simplify_path.
    Returns the simplified path goal first (as mirte_rrt gives it), or None."""
    start = np.array([pos[0], pos[1]])
    goal = np.array(GOAL)
    # Same LocalMap as Ex4 (box radius 0.20 m + MIRTE radius 0.22 m), the boxes as landmarks in the world frame
    robot = robot_models.PointMassModel(ctrl_range=[-RRT_PATH_RES, RRT_PATH_RES])
    for _ in range(RRT_TRIES): # RRT is random, try again if it does not find a path
        rrt = RRT(start=start, goal=goal, robot_model=robot, map=m,
                expand_dis=RRT_EXPAND_DIS, path_resolution=RRT_PATH_RES)
        path = rrt.planning(animation=False)
        if path is not None:
            return simplify_path(path, rrt)
    return None

def create_local_map(local_landmarks, local_obstacles, dist) -> local_map.LocalMap: #TODO
    lm = local_map.LocalMap(low=(-100, 0.0),
    high=(100, min(dist+50,350)), landmark_radius=20,
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
                    x + delta_x * np.cos(theta) - delta_y * np.sin(theta),
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
                        x + delta_x * np.cos(theta) - delta_y * np.sin(theta),
                        y + delta_x * np.sin(theta) + delta_y * np.cos(theta)])
                    if is_outisde_map(pos, map.obstacale_radius, map.map_area):
                        #obstacle can safely be added as it is outise the map
                        obstales[id] = pos
                        continue
                return True
    return False

### Updating MCL ###
def update_mcl(mcl, u, z, global_map, NUM_PARTICLES) -> None:
    "function for updating MCL"
    if z:
        mcl.mcl(u,z,global_map)
    else: #TODO look at this
        if u[0] != 0.0 or u[1] != 0.0: # without a move or a measurement nothing changed (and the motion model adds noise for u = 0)
            mcl.particles = np.array([mcl.sample_motion_model_with_map(u, x_last, global_map) for x_last in mcl.particles])
        for p in mcl.particles:
            p.setWeight(1.0/NUM_PARTICLES)


u_pending = [0.0, 0.0]
z = {}
seen_landmark_ids = {}
seen_obstacle_ids = {}

#while stack run or target run the loop
while target_stack or target:
    #if we dont have a target then pop from the target stack
    if not target:
        #pop target method
        target = pop(target_stack, global_map)
        print(f"New target has been assigned: {target.id}")
        continue

    #get u 
    u = u_pending

    #get observations
    z, local_landmarks, local_obstacles = cam.get_observations(landmarks, mirte)

    #print information as to what the mirte sees in local view
    cam.print_observations(z, local_landmarks, local_obstacles)

    #update mcl
    update_mcl(aug_mcl, u, z, global_map, NUM_PARTICLES)
    
    est_pose = aug_mcl.estimate_pose() # The estimate of the robots current pose

    #have we reached convergence of paricle pose
    has_converged = aug_mcl.has_converged(est_pose, aug_mcl, X_VAR_ERR, Y_VAR_ERR, THETA_VAR_ERR)

    #print information regarding estimated pose and convergence
    print(f"estimated pose: X = {est_pose.getX()}, Y = {est_pose.getY()}, Theta = {est_pose.getTheta()}\nConvergence: {has_converged}")

    #guess_conversion =  guess_conversion_func(last_usefull_landmarkid, control_actions, local_landmarks, local_obstacles)

    #have the mirte reached the target?
    reached_target = has_reached_target(target, z, est_pose)

    if reached_target:
        #TODO add print information as to we beilve we reached target
        print("target beleived to have been reched")
        reset_turn_variables()
        target, previous_target = target_update(target, previous_target)
        if target:
            raise ValueError('target has not been set to None')
        u_pending = [0.0, 0.0]
        continue

    ### control flow of decisions ###
    if target.id in local_landmarks: #if the mirte sees the target then create and excute a path plan
        #TODO create rrt path as close to target as possible
        dist = z[target.id][0]
        Local_MAP = create_local_map(local_landmarks, local_obstacles, dist)

        #TODO fix goal calculation
        #TODO handle angles
        goal = (local_landmarks[target.id][0],local_landmarks[target.id][1]) # TODO remove 50(20 + 30) cm from dist

        #TODO init this
        target_run = ...

        robot = robot_models.PointMassModel(ctrl_range=[-RRT_PATH_RES, RRT_PATH_RES])


        #TODO wrap this in a method
        rrt = RRT(start=[0.0], goal=goal, robot_model=robot, map=Local_MAP,
                            expand_dis=RRT_EXPAND_DIS, path_resolution=RRT_PATH_RES)
        path = None

        size = ... 
        for _ in range(RRT_TRIES): # RRT is random, try again if it does not find a path
            #TODO if size of rrt isnt big enough increase size of local map
            #TODO we want the robot to use as little space as possible in a rectangle, but increase if no path is found
            rrt = RRT(start=[0.0], goal=goal, robot_model=robot, map=Local_MAP,
                                        expand_dis=RRT_EXPAND_DIS, path_resolution=RRT_PATH_RES)
            path = rrt.planning(animation=False)
            if path is not None:
                path = simplify_path(path, rrt)
                break
            size += ...

        while path is None:
            path = rrt_plan([0,0, 0.0], Local_MAP, goal)
            #could mirte find a proper plan with current goal or should it be updated
            if path is None:#TODO update goal
                #update goal with some random setting in vacinity
                print('could not find path, so trying with goal in nearby vacinity')
                target_run = False
                goal = ...


        reached_target = drive.Execute_path(path, 
                                      [0.0, 0.0, np.radians(90)], #we are looking foward in local planning so add 90 degrees
                                      mirte, 
                                      aug_mcl, 
                                      local_landmarks, 
                                      local_obstacles, 
                                      landmarks, 
                                      Local_MAP,
                                      target,
                                      global_map, 
                                      target_run, 
                                      False)

        reset_turn_variables()
        if reached_target:
            print("target beleived to have been reached, via local pathplanning")
            target, previous_target = target_update(target, previous_target)
            if target:
                raise ValueError('target has not been set to None')
        u_pending = [0.0, 0.0]
        continue
    elif has_converged: #if the mirte has a good idea of estimated pose then create and excute a path plan to the target
        #then do planning
        #TODO do presteps

        #TODO init this properly. Is it always a run towards the target?
        target_run = True

        reached_target = drive.Execute_path(path, 
                                      [est_pose.getX(), est_pose.getY(), est_pose.getTheta()], 
                                      mirte, 
                                      aug_mcl, 
                                      local_landmarks, 
                                      local_obstacles,
                                      landmarks, 
                                      global_map, 
                                      global_map,
                                      target, 
                                      True, 
                                      True)
        
        reset_turn_variables()
        if reached_target:
            print("target beleived to have been reached, via global pathplanning")
            target, previous_target = target_update(target, previous_target)
            if target:
                raise ValueError('target has not been set to None')
        u_pending = [0.0, 0.0]
        continue
    #elif guess_conversion: #TODO
        #we have now split the potential maps in half atleast, so do some action
        #pass

    elif not (HAS_TURNED_LEFT and HAS_TURNED_RIGHT):
        #left right step
        #turn 20 degrees to the left
        if not HAS_TURNED_LEFT:
            HAS_TURNED_LEFT = True
            drive.turn(SCAN_STEP, mirte)
            u_pending = [0.0, SCAN_STEP]
            continue
        #turn 40 degrees to the right
        if not HAS_TURNED_RIGHT:
            HAS_TURNED_RIGHT = True
            drive.turn(-2*SCAN_STEP, mirte)
            u_pending = [0.0, -2*SCAN_STEP]
            continue
        raise ValueError('somthing went wrong in has turned left and has turned right')
    elif turns <  MAX_TURNS: #mirte see anything, so scan in a circle to see if it gets bearings
        #360
        turns += 1
        #turn 20 degrees to the right
        drive.turn(-SCAN_STEP, mirte)
        u_pending = [0.0, -SCAN_STEP]
    else: #TODO #mirte could not gather anything from this position, do some random exploration in the close vacinity
        #TODO add random exploration
        #rember to add 90 degrres when executing local path
        #random exploration
        reset_turn_variables()
        pass


    #can we see target?

        #are there objects in the way?

    #can we see any landmarks?
    
print(f"No targets left, shutdown inbound")
    
#done
del  mirte

    