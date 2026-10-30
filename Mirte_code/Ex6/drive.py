
import camera as cam
import numpy as np
from mcl import MCL
from planning import should_replan
import time


###DRVING CONSTANTS
#TODO refine constants on charged battery
LINEAR_OFFSET = -0.0354
LINEAR_SPEED = 0.3
ANGULAR_SPEED = 0.7
LEFT_TRUN_SCALER = 1.05 
RIGHT_TRUN_SCALER = 1.1
SAFE_DISTANCE_BOTH = 0.35 #if both go off then ahead of/behind mirte 
SAFE_DISTANCE_ONE = 0.20 #if one go off then at diagonal of mirte 
BACKSTEP_SIZE = 0.4 #m
DRIVE_TIME_OFFSET = 0.2 #offset slept in KU_MIRTE drive method
#### DRIVING METHODS ####
def turn(theta, mirte) -> float:
    "Turning method, returns turned amount"
    if abs(theta) > 1e-3:
        scaler = LEFT_TRUN_SCALER if theta > 0 else RIGHT_TRUN_SCALER
        mirte.drive(
            0.0,
            np.sign(theta) * ANGULAR_SPEED,
            scaler * abs(theta) / ANGULAR_SPEED
            )
        return theta
    return 0.0

def obstacle_avoidance_front(mirte) -> bool:
    "method checking if we need to stop do to obstacle"
    # Read sonar sensors
    sonar = mirte.sonar

    left = sonar["front_left"]
    right = sonar["front_right"]

    print(
        f"Left sonar dist: {left*100:.2f} cm | "
        f"Right sonar dist: {right*100:.2f} cm"
    )
    
            # No obstacle
    if left <= SAFE_DISTANCE_BOTH and right <= SAFE_DISTANCE_BOTH:
        print('obstacle detected')
        return True
    if( left <= SAFE_DISTANCE_ONE) or (right <= SAFE_DISTANCE_ONE):
            print('obstacle detected')
            return True
    return False

def obstacle_avoidance_rear(mirte) -> bool:
    "method checking if we need to stop do to obstacle"
    # Read sonar sensors
    sonar = mirte.sonar

    left = sonar["rear_left"]
    right = sonar["rear_right"]

    print(
        f"Left rear sonar dist: {left*100:.2f} cm | "
        f"Right rear sonar dist: {right*100:.2f} cm"
    )
    
            # No obstacle
    if left <= SAFE_DISTANCE_BOTH and right <= SAFE_DISTANCE_BOTH:
        print('obstacle detected')
        return True
    if( left <= SAFE_DISTANCE_ONE) or (right <= SAFE_DISTANCE_ONE):
            print('obstacle detected')
            return True
    return False



def drive_forward(distance, mirte, in_meters=False) -> float: #TODO make use of obstacle avoidance
    # Convert from cm to meters
    if not in_meters:
        distance = distance / 100

    cant_drive = obstacle_avoidance_front(mirte)
    if cant_drive:
        return 0.0, False

    start = time.perf_counter()
    drive_time = distance / LINEAR_SPEED
    
    mirte.drive(
        LINEAR_SPEED,
        LINEAR_OFFSET,
        drive_time,
        blocking=False
    )

    while time.perf_counter() - start < drive_time -DRIVE_TIME_OFFSET:
        cant_drive = obstacle_avoidance_front(mirte)
        if cant_drive:
            driven_dist = ((time.perf_counter() - start + DRIVE_TIME_OFFSET)/drive_time)*distance #TODO check if this needs updating
            mirte.stop()
            return driven_dist, False

        time.sleep(0.05)

    return distance, True


def drive_backward(distance, mirte, in_meters=False) -> float: #TODO make use of obstacle avoidance
    # Convert from cm to meters
    if not in_meters:
        distance = distance / 100

    cant_drive = obstacle_avoidance_rear(mirte)
    if cant_drive:
        return 0.0, False

    start = time.perf_counter()
    drive_time = distance / LINEAR_SPEED

    #reverse mirte
    mirte.drive(
        -LINEAR_SPEED,
        -LINEAR_OFFSET,
        drive_time,
        blocking=False
    )

    while time.perf_counter() - start < drive_time -DRIVE_TIME_OFFSET:
        cant_drive = obstacle_avoidance_rear(mirte)
        if cant_drive:
            driven_dist = ((time.perf_counter() - start + DRIVE_TIME_OFFSET)/drive_time)*distance #TODO check if this needs updating
            mirte.stop()
            return driven_dist, False

        time.sleep(0.05)

    return distance, True


def Execute_path(path, pose, mirte, mcl, local_landmarks, local_obstacles, landmarks, map, global_map, target, target_run, global_plan = False) -> bool:
    "method for executing path"
    current_pose = np.array(pose, dtype=float).copy()
    u = [0.0, 0.0]
    #TODO be aware i changed from path.revsere() to this, that could make trouble
    list_path = list(reversed(path))[1:]
    for point in list_path:
        dx = point[0] - current_pose[0]
        dy = point[1] - current_pose[1]

        distance = np.linalg.norm(np.array([dx, dy]))/100

        if distance <= 1e-6:
            u = [0.0, 0.0]
            continue

        target_theta = np.arctan2(dy, dx)

        dtheta = target_theta - current_pose[2]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))


        #turn
        turned = turn(dtheta, mirte)


        #update current pose
        current_pose[2] = np.mod(target_theta, 2.0 * np.pi)
        u = [0.0, turned]

        print(f"Distance turned: {turned}")

        #get_observations
        new_z, new_local_landmarks, new_local_obstacles = cam.get_observations(landmarks, mirte)

        
        #print observations
        cam.print_observations(new_z, new_local_landmarks, new_local_obstacles)
        
        #replan check
        replan = should_replan(local_landmarks, local_obstacles, new_local_landmarks, new_local_obstacles, map, current_pose, target, global_plan)

        if replan:
            print('new observation made, replanning in order')
            MCL.update_mcl(mcl, u, new_z, global_map, len(mcl.particles))
            return False
            

        #TODO add obstavle avoince during drive with sonar
        dist_driven, can_safely_drive = drive_forward(distance, mirte, in_meters= True)
        if not can_safely_drive:
            dist_forward = dist_driven
            #try to move back 40 cm if obstacle ahead
            dist_driven -= drive_backward(BACKSTEP_SIZE, mirte, in_meters= True)
            #if close and last point in path then we assume we reached a target obstacle
            if (distance - dist_forward) > 0.30 or point is not list_path[-1]:
                #get obsercations
                new_z, new_local_landmarks, new_local_obstacles = cam.get_observations(landmarks, mirte)
                        
                #print observations
                cam.print_observations(new_z, new_local_landmarks, new_local_obstacles)
                u = [dist_driven * 100.0, turned]
                print(f"Distance driven: {dist_driven}, distance turned: {turned}")
                MCL.update_mcl(mcl, u, new_z, global_map, len(mcl.particles))
                return False

        #get_observations
        new_z, new_local_landmarks, new_local_obstacles = cam.get_observations(landmarks, mirte)
        
        #print observations
        cam.print_observations(new_z, new_local_landmarks, new_local_obstacles)

        #update mcl
        u = [dist_driven * 100.0, turned]
        print(f"Distance driven: {dist_driven}, distance turned: {turned}")
        MCL.update_mcl(mcl, u, new_z, global_map, len(mcl.particles))
       
        
        #return False
        
        #replan check
        replan = should_replan(local_landmarks, local_obstacles, new_local_landmarks, new_local_obstacles, map, current_pose, target, global_plan)

        if replan:
            print('new observation made, replanning in order')
            return False

        # Update predicted pose without modifying the input pose.
        current_pose[0] = point[0]
        current_pose[1] = point[1]
    #whether we did random exploration or if we are drivng towards the target
    if target_run:
        return True
    return False