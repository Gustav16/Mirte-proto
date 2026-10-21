#import packages
import local_map
from mcl import MCL
import particle as pcl
import camera as cam
from rrt import RRT, simplify_path, Execute_path
import robot_models



from ku_mirte import KU_Mirte
#init landmark
landmarkIDs = [1,2,3,4]

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

#init landmarks
landmarks = {
    landmarkIDs[0]: (0.0, 0.0),
    landmarkIDs[1]: (0.0, 300.0),
    landmarkIDs[2]: (400.0, 0.0),  
    landmarkIDs[3]: (300.0, 400.0)  
}


#init map
global_map = local_map.LocalMap(low=(-120.0, -120.0),
        high=(420.0, 520.0), landmark_radius=20,
                mirte_radius=22) #TODO update local map so it can set radius for different landmarks

Local_MAP = None

global_map.landmarks = [[list(landmarks[ID]), 20.0 + 40.0, ID] for ID in landmarkIDs] # same landmarks as above, in the map's format
#update local map for our local map that acccounts for obstacles


#init needed objects and constants

#number of particles 
NUM_PARTICLES = 1000

#init particles
particles = pcl.initialize_particles(NUM_PARTICLES,global_map) #init prior

#init mcl r
aug_mcl = MCL(particles)

#init mirte
mire = KU_Mirte()

#convergence constants
X_VAR_ERR = 30**2
Y_VAR_ERR = 30**2
THETA_VAR_ERR = 20**2


#RRT init 
RRT_PATH_RES = 10      # cm
RRT_EXPAND_DIS = 40    # cm
RRT_MAP_MARGIN = 0.5    # m, free space around start, goal and boxes for the RRT samples
RRT_TRIES = 5


def pop(target_stack, m) -> Target:
    "returns the new target"
    if target_stack:
        new_target =  target_stack.pop()
        m.update_milestone_radius(new_target.id, 20)
        return new_target
    return ValueError('stack empty but pop was called')
        
def target_update(target, previous_target, m) -> None:
    "method for updating target"
    if previous_target:
        m.update_milestone_radius(previous_target.id, 60)
    m.update_milestone_radius(target.id, 20)
    #set values
    previous_target = target
    target = None


def has_reached_target(m):
    pass

u_pending = [0.0, 0.0]
z = {}
seen_landmark_ids = {}
seen_obstacle_ids = {}

#while stack run loop
while target_stack or target:
    if not target:
        #pop target method
        target = pop(target_stack, target, global_map)
        continue
    #u, z, 

    #get u 
    u = u_pending


    #get observation z (get camera image)  update z for landmarks
    obs = {}

    #obstacles are for local path planning
    obstacles = {}

    #change this to better take in camera offset and

    #for landmarks wee just need the angle to center and 

    z,local_landmarks, local_obstacles = cam.get_observations(landmarks, mirte)

    #update mcl
    if z:
        aug_mcl.mcl(u,z,global_map)
    else: #TODO look at this
        if u[0] != 0.0 or u[1] != 0.0: # without a move or a measurement nothing changed (and the motion model adds noise for u = 0)
            aug_mcl.particles = np.array([aug_mcl.sample_motion_model_with_map(u, x_last, global_map) for x_last in aug_mcl.particles])
        for p in aug_mcl.particles:
            p.setWeight(1.0/NUM_PARTICLES)

    
    
    est_pose = aug_mcl.estimate_pose() # The estimate of the robots current pose
    
    has_converged = aug_mcl.has_converged(est_pose, aug_mcl, X_VAR_ERR, Y_VAR_ERR, THETA_VAR_ERR)

    guess_conversion =  guess_conversion_func(last_usefull_landmarkid, control_actions, local_landmarks, local_obstacles)

    def rrt_plan(pose, m):
        """Plan in the world frame (m) from MIRTE's estimated position (from the measurements and motion updates of the
        particle filter) to GOAL around the boxes with mirte_rrt's RRT, then shorten it with simplify_path.
        Returns the simplified path goal first (as mirte_rrt gives it), or None."""
        start = np.array([pose[0], pose[1]])
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

    def create_local_map(local_landmarks, local_obstacles, dist):
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
            obs.append([pos, id])
        lm.landmarks = lmarks
        lm.obstacles = lobs
        return lm


    #do control flow of decision
    if target.id in local_landmarks:
        ##then do somthing
        #create rrt path as close to target as possible
        dist = z[target.id][0]
        Local_MAP = create_local_map(local_landmarks, local_obstacles, dist)

        #TODO handle angles
        
        goal = (local_landmarks[target.id][0],local_landmarks[target.id][1]) # remove 30 cm from dist

        robot = robot_models.PointMassModel(ctrl_range=[-RRT_PATH_RES, RRT_PATH_RES])



        rrt = RRT(start=[0.0], goal=goal, robot_model=robot, map=Local_MAP,
                            expand_dis=RRT_EXPAND_DIS, path_resolution=RRT_PATH_RES)

        


        #TODO if size of rrt isnt big enough increase sze 
        pass
    elif has_converged:
        #then do planning
        pass
    elif guess_conversion:
        #we have now split the potential maps in half atleast, so do some action
        pass

    elif not (HAS_TURNED_LEFT and HAS_TURNED_RIGHT):
        #left right step
        pass
    elif turns <  MAX_TURNS:
        #360
        turns += 1
        #TODO rotate turn angle
        pass
    else:
        #random exploration
        pass


    #can we see target?

        #are there objects in the way?

    #can we see any landmarks?

    
    #at target, shange target
        #target_update(target, previous_target)
    
        
    



#done
#del  mirte

    