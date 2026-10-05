"""Segment follower: rotate in place, then short blocking translations.
The physical path stays on the planned polyline rather than cutting corners.
"""
import math
import time
import numpy as np
from mcl import wrap_angle
from robot_io import move,stop
import ex5_config as cfg


def closest_path_index(path,pose):
    return int(np.argmin([np.linalg.norm(np.asarray(p)-pose[:2]) for p in path]))


def lookahead_point(path,pose,lookahead=.25):
    index=closest_path_index(path,pose);d=0.0
    for i in range(index,len(path)-1):
        d+=np.linalg.norm(np.asarray(path[i+1])-path[i])
        if d>=lookahead:return np.asarray(path[i+1],float)
    return np.asarray(path[-1],float)


def follow_path(path,mirte,localizer=None,get_observations=None,landmarks=None,
                linear_speed=.22,max_angular_speed=.65,lookahead=.25,
                goal_tolerance=.06,control_period=.25,heading_gain=1.8,max_time=120,
                sonar_stop_distance=.30,sonar_slow_distance=.45,world_map=None):
    # Compatibility keywords remain accepted; execution follows every vertex.
    if path is None or len(path)<2:return False
    path=[np.asarray(p,float) for p in path]
    pose=np.array([*path[0],0.]) if localizer is None else localizer.estimate_pose()
    started=time.monotonic();unobserved=0.0
    try:
        for index,target in enumerate(path[1:],1):
            tolerance=goal_tolerance if index==len(path)-1 else .025
            while np.linalg.norm(target-pose[:2])>tolerance:
                if time.monotonic()-started>=max_time:return False
                dx,dz=target-pose[:2];error=float(wrap_angle(math.atan2(-dx,dz)-pose[2]))
                if abs(error)>math.radians(5):
                    ds=0.;da=float(np.clip(error,-cfg.MAX_ROTATION_STEP_RAD,cfg.MAX_ROTATION_STEP_RAD))
                else:
                    ds=min(float(np.hypot(dx,dz)),.08);da=0.
                    endpoint=pose[:2]+np.array([-math.sin(pose[2]),math.cos(pose[2])])*ds
                    if world_map is not None and not world_map.segment_is_free(pose[:2],endpoint):
                        print('Predicted translation intersects frozen map. Replan required.');return False
                movement=move(mirte,ds,da,linear_speed,max_angular_speed,sonar_stop_distance)
                if localizer is not None:
                    localizer.predict(*movement)
                    time.sleep(cfg.CAMERA_SETTLE_SECONDS)
                    obs=[] if get_observations is None else get_observations()
                    valid=localizer.valid_observations(obs)
                    localizer.correct(valid);localizer.resample();pose=localizer.estimate_pose()
                    unobserved=0.0 if valid else unobserved+ds
                    if unobserved>cfg.PATH_MAX_UNOBSERVED_DISTANCE_M or not localizer.confident(.10,math.radians(15)):
                        print('Path localization unavailable/uncertain. Stop and replan.');return False
                else:
                    pose[2]=wrap_angle(pose[2]+da);pose[0]-=math.sin(pose[2])*ds;pose[1]+=math.cos(pose[2])*ds
        print('Path endpoint reached within estimated tolerance.');return True
    finally:stop(mirte)
