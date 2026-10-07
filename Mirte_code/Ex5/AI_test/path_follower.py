"""Continuous, slow path following with MCL and predicted-arc collision checks."""
import math
import time
import numpy as np
from mcl import wrap_angle
from robot_io import require_continuous_api,front_clearance,stop
from continuous_control import MotionTracker,arc_is_free
import ex5_config as cfg


def closest_path_index(path,pose):
    return int(np.argmin([np.linalg.norm(np.asarray(p)-pose[:2]) for p in path]))


def path_target(path,pose,lookahead,minimum_segment=0):
    """Project onto the polyline, then advance by arc length; never go backwards."""
    best=None
    for i in range(minimum_segment,len(path)-1):
        a,b=path[i],path[i+1];d=b-a;length2=float(d@d)
        t=0. if length2<1e-12 else float(np.clip((pose[:2]-a)@d/length2,0,1))
        p=a+t*d;distance=float(np.linalg.norm(p-pose[:2]))
        if best is None or distance<best[0]:best=(distance,i,p)
    _,index,projection=best
    remaining=lookahead;current=projection
    for i in range(index,len(path)-1):
        delta=path[i+1]-current;length=float(np.linalg.norm(delta))
        if length>=remaining and length>1e-12:return current+delta*(remaining/length),index
        remaining-=length;current=path[i+1]
    return path[-1],index


def lookahead_point(path,pose,lookahead=.15):
    return path_target([np.asarray(p,float) for p in path],pose,lookahead)[0]


def follow_path(path,mirte,localizer=None,get_observations=None,landmarks=None,
                linear_speed=.08,max_angular_speed=.45,lookahead=.15,
                goal_tolerance=.06,control_period=.15,heading_gain=1.5,max_time=180,
                sonar_stop_distance=.30,sonar_slow_distance=.45,world_map=None):
    if path is None or len(path)<2:return False
    path=[np.asarray(p,float) for p in path]
    if any(p.shape!=(2,) or not np.all(np.isfinite(p)) for p in path):raise ValueError('Invalid path.')
    tracker=None
    try:
        require_continuous_api(mirte)
        tracker=MotionTracker(mirte,localizer,pose=[*path[0],0.])
        if localizer is not None:tracker.pose=localizer.estimate_pose()
        started=time.monotonic();last_seen_distance=0.;segment=0
        for tick in range(cfg.SMOOTH_MAX_TICKS):
            tick_start=time.monotonic()
            if tick_start-started>max_time:return False
            observations=[] if get_observations is None else get_observations()
            tracker.advance()  # predict old velocity to observation receipt time
            if localizer is not None:
                valid=localizer.valid_observations(observations)
                localizer.correct(valid);localizer.resample();tracker.pose=localizer.estimate_pose()
                if valid:last_seen_distance=tracker.distance
                if tracker.distance-last_seen_distance>cfg.PATH_MAX_UNOBSERVED_DISTANCE_M or not localizer.confident(.10,math.radians(15)):
                    print('Localization unavailable or uncertain. Continuous drive stopped.');return False
            pose=tracker.pose;distance=float(np.linalg.norm(path[-1]-pose[:2]))
            if distance<=goal_tolerance:
                print('Path endpoint reached.');return True
            clearance=front_clearance(mirte,sonar_stop_distance)
            if sonar_stop_distance is not None and clearance<=sonar_stop_distance:
                print('Sonar safety stop.');return False
            # Use a closer target if a corner-cutting arc would hit a box.
            candidate=None
            for ahead in (lookahead,lookahead/2,.03):
                target,new_segment=path_target(path,pose,ahead,segment)
                dx,dz=target-pose[:2];error=float(wrap_angle(math.atan2(-dx,dz)-pose[2]))
                w=float(np.clip(heading_gain*error,-max_angular_speed,max_angular_speed))
                v=min(linear_speed,.6*distance)*max(0.,math.cos(error))**2
                if abs(error)>math.radians(75):v=0.
                if clearance<sonar_slow_distance:v=min(v,.04)
                if arc_is_free(world_map,pose,v,w,cfg.SMOOTH_COLLISION_HORIZON_S):
                    candidate=(v,w,new_segment);break
            if candidate is None:
                target,new_segment=path_target(path,pose,.03,segment)
                error=float(wrap_angle(math.atan2(-(target[0]-pose[0]),target[1]-pose[1])-pose[2]))
                candidate=(0.,float(np.clip(heading_gain*error,-max_angular_speed,max_angular_speed)),new_segment)
                if abs(candidate[1])<.01:
                    print('No collision-free continuous command. Replan required.');return False
            desired_v,desired_w,segment=candidate
            v,w=tracker.slew(desired_v,desired_w,control_period,cfg.SMOOTH_ACCELERATION_M_S2,cfg.SMOOTH_ANGULAR_ACCELERATION_RAD_S2)
            if not arc_is_free(world_map,pose,v,w,cfg.SMOOTH_COLLISION_HORIZON_S):
                # Safety deceleration overrides the normal smoothing limit.
                v=0.
            tracker.command(v,w)
            elapsed=time.monotonic()-tick_start
            if elapsed<control_period:time.sleep(control_period-elapsed)
        print('Control tick budget exhausted.');return False
    finally:stop(mirte)
