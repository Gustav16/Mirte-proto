#import packages
import local_map
from mcl import MCL
import particle as pcl
import camera as cam
import numpy as np
import time
import drive
import planning
import mirte_visualizer as vis

from ku_mirte import KU_Mirte
#init landmark
landmarkIDs = [6,3,1,4]


#init landmarks
landmarks = {
    landmarkIDs[0]: (0.0, 0.0),
    landmarkIDs[1]: (0.0, 300.0),
    landmarkIDs[2]: (400.0, 0.0),  
    landmarkIDs[3]: (400.0, 300.0)  
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
        high=(520.0, 420.0), landmark_radius=LANDMARK_RADIUS,
                mirte_radius=22) #TODO update local map so it can set radius for different landmarks

Local_MAP = None

global_map.landmarks = [[list(landmarks[ID]), 20.0 + 40.0, ID] for ID in landmarkIDs] # same landmarks as above, in the map's format
#update local map for our local map that acccounts for obstacles

#init needed objects and constants

# VISUALIZER

vis.start(globals(), period=5.0, web=True)

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
X_VAR_ERR = 25**2 #reduced from 30 to 25
Y_VAR_ERR = 25**2  #reduced from 30 to 25
THETA_VAR_ERR = np.radians(15)**2 #reduced from 20 to 15


#goal update constants
MAX_GOAL_ATTEMPTS = 10
GOAL_SEARCH_STEP = 20 # cm moved along the circle around the target per step

#Turn variables
HAS_TURNED_LEFT = False
HAS_TURNED_RIGHT = False
turns = 0
has_turned_random_exploration = False

#max turn constant
SCAN_STEP = np.radians(30)
MAX_TURNS = (np.radians(360)//SCAN_STEP) - 2


### Method for reseting turning varibles ###
def reset_turn_variables() -> None:
    "method for reseting varibles used in turning scheme"
    global HAS_TURNED_LEFT, HAS_TURNED_RIGHT, turns, has_turned_random_exploration

    HAS_TURNED_LEFT = False
    HAS_TURNED_RIGHT = False
    turns = 0
    has_turned_random_exploration = False


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

    robot_pos = np.array([est_pose.getX(), est_pose.getY()])
    target_pos = np.array(landmarks[id])

    distance_to_target = np.linalg.norm(robot_pos - target_pos)

    if has_converged and distance_to_target <= SUCCESS__RANGE:
        return True
        
    #TODO check if there are other success conditions
    return False




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
    MCL.update_mcl(aug_mcl, u, z, global_map, NUM_PARTICLES)
    
    est_pose = aug_mcl.estimate_pose() # The estimate of the robots current pose

    #have we reached convergence of paricle pose
    has_converged = MCL.has_converged(est_pose, aug_mcl, X_VAR_ERR, Y_VAR_ERR, THETA_VAR_ERR)

    #print information regarding estimated pose and convergence
    print(f"estimated pose: X = {est_pose.getX()}, Y = {est_pose.getY()}, Theta = {est_pose.getTheta()}\nConvergence: {has_converged}")

    #guess_conversion =  guess_conversion_func(last_usefull_landmarkid, control_actions, local_landmarks, local_obstacles)

    #have the mirte reached the target?
    reached_target = has_reached_target(target, z, est_pose, has_converged)

    if reached_target:
        print(f"target: {target.id} beleived to have been reched")
        reset_turn_variables()
        target, previous_target = target_update(target, previous_target, global_map)
        if target:
            raise ValueError('target has not been set to None')
        u_pending = [0.0, 0.0]
        continue

    ### control flow of decisions ###
    if target.id in local_landmarks: #if the mirte sees the target then create and excute a path plan
        
        #TODO create rrt path as close to target as possible
        dist = z[target.id][0]
        print(f"Target: {target.id} seen at distance {dist}. Creating local path")

        max_dist = min(dist, 201)
        Local_MAP = planning.create_local_map(local_landmarks, local_obstacles, max_dist, previous_target, target)
        goal_len = min(abs(max_dist - 60), abs(max_dist -30) ) if dist <= 200 else max_dist
        len = goal_len/ dist
        goal = (local_landmarks[target.id][0]*len,local_landmarks[target.id][1]*len)

        #TODO init this
        target_run = z[target.id][0] <= 200

        path = None
        target_pos = np.array(local_landmarks[target.id])
        goal_offset = np.array(goal) - target_pos #vector from target to the original goal
        goal_radius = np.linalg.norm(goal_offset)
        attempts = 0

        while path is None and attempts < MAX_GOAL_ATTEMPTS:

            if not Local_MAP.in_collision(goal): # Check if goal is inside obstacle
                path = planning.rrt_plan([0,0, 0.0], Local_MAP, goal, True) # If not do rrt planning
            
            if path is None: # If not path is found update goal
                attempts += 1
                print('could not find path, so trying with goal in nearby vacinity')
                target_run = False
                step = (attempts + 1) // 2 * (1 if attempts % 2 else -1) # Calculate step: 1, -1, 2, -2, 3, -3...
                angle = step * GOAL_SEARCH_STEP / goal_radius # Calculate angle if vector 
                rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle),  np.cos(angle)]]) # Rotate vector
                goal = target_pos + rotation @ goal_offset # Update goal


        reached_target = drive.Execute_path(path, 
                                      [0.0, 0.0, np.radians(90)], #we are looking foward in local planning so add 90 degrees
                                      mirte, 
                                      aug_mcl, 
                                      local_landmarks, 
                                      local_obstacles, 
                                      landmarks, 
                                      Local_MAP,
                                      global_map,
                                      target,
                                      target_run, 
                                      False)

        reset_turn_variables()
        if reached_target:
            print("target beleived to have been reached, via local pathplanning")
            target, previous_target = target_update(target, previous_target, global_map)
            if target:
                raise ValueError('target has not been set to None')
        u_pending = [0.0, 0.0]
        continue

    elif drive.obstacle_avoidance_front(mirte) and  (not drive.obstacle_avoidance_rear(mirte)): # if somthing in front of mirte
        dist_driven,_ = drive.drive_backward(drive.BACKSTEP_SIZE, mirte, in_meters= True) #try to drive backwards 40 cm
        u_pending = [-dist_driven*100, 0.0] #set control action for update
            
    elif has_converged: #if the mirte has a good idea of estimated pose then create and excute a path plan to the target
        #then do planning
        #TODO do presteps

        #dist = z[target.id][0]
        #Local_MAP = planning.create_local_map(local_landmarks, local_obstacles, dist)

        #update obstacles #TODO not sure
        #update the local obstacles to gloabal obstacles
        temp_global_map = global_map.Copy()

        temp_global_map.obstacles = [( np.array([
            est_pose.getX() + y * np.cos(est_pose.getTheta()) + x * np.sin(est_pose.getTheta()),
            est_pose.getY() + y * np.sin(est_pose.getTheta()) - x * np.cos(est_pose.getTheta())

            #est_pose.getX() + x * np.cos(est_pose.getTheta()) - y * np.sin(est_pose.getTheta()),
            #est_pose.getY() + x * np.sin(est_pose.getTheta()) + y * np.cos(est_pose.getTheta())
            ]),
            id
        )
        for id, (x, y) in local_obstacles.items()
]
        target_pos = list(landmarks[target.id])

        delta = target_pos - np.array([est_pose.getX(), est_pose.getY()])

        dist = np.linalg.norm(delta)
        max_dist = min(dist, 201)
        goal_len = min(abs(max_dist - 60), abs(max_dist -30) ) if dist <= 200 else max_dist
        length = goal_len/ dist

        goal = np.array([est_pose.getX(), est_pose.getY()]) + delta * length

        target_run = False

        path = None

        while path is None:
            path = planning.rrt_plan([est_pose.getX(), est_pose.getY()], temp_global_map, goal, False)
            #could mirte find a proper plan with current goal or should it be updated
            if path is None:#TODO update goal
                #update goal with some random setting in vacinity
                print('could not find path, so trying with goal in nearby vacinity')
                target_run = False
                goal = goal

        reached_target = drive.Execute_path(path, 
                                      [est_pose.getX(), est_pose.getY(), est_pose.getTheta()], 
                                      mirte, 
                                      aug_mcl, 
                                      local_landmarks, 
                                      local_obstacles,
                                      landmarks, 
                                      temp_global_map, 
                                      global_map,
                                      target, 
                                      True, 
                                      True)
        #reset global obstacles
        reset_turn_variables()
        if reached_target:
            print("target beleived to have been reached, via global pathplanning")
            target, previous_target = target_update(target, previous_target, global_map)
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

    else:
        if not has_turned_random_exploration:
            angle = np.random.uniform(-np.pi, np.pi)
            turned = drive.turn(angle, mirte)
            has_turned_random_exploration = True
            u_pending = [0.0, turned]
            continue

        Local_MAP = planning.create_local_map(local_landmarks, local_obstacles, 150, previous_target, target)
        goal = (0.0, 150.0)
        path = planning.rrt_plan([0, 0, 0.0], Local_MAP, goal, True)

        if path is not None:
            drive.Execute_path(path,
                               [0.0, 0.0, np.radians(90)], #we are looking foward in local planning so add 90 degrees
                               mirte,
                               aug_mcl,
                               local_landmarks,
                               local_obstacles,
                               landmarks,
                               Local_MAP,
                               global_map,
                               target,
                               False,
                               False)
        else:
            print('could not find exploration path')

        reset_turn_variables()
        u_pending = [0.0, 0.0]

    #can we see target?
    #are there objects in the way?
    #can we see any landmarks?
print(f"No targets left, shutdown inbound")
    
#done
del  mirte

    