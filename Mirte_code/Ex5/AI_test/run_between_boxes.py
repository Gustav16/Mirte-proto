"""Exercise 5: MCL in a known marker frame, with stationary scan/recovery.
Motor commands are dead-reckoned; physical calibration is still required.
"""
import math
import time
import numpy as np
import ex5_config as cfg
from mcl import MCL,wrap_angle
from aruco_measurements import observe_mirte
from robot_io import KU_Mirte,move,stop


def rotate_step(mirte,angle):
    angle=float(np.clip(angle,-cfg.MAX_ROTATION_STEP_RAD,cfg.MAX_ROTATION_STEP_RAD))
    return move(mirte,0,angle,cfg.LINEAR_SPEED,cfg.ANGULAR_SPEED,cfg.SONAR_STOP_DISTANCE_M)


def translate_step(mirte,distance):
    distance=float(np.clip(distance,0,cfg.MAX_TRANSLATION_STEP_M))
    return move(mirte,distance,0,cfg.LINEAR_SPEED,cfg.ANGULAR_SPEED,cfg.SONAR_STOP_DISTANCE_M)


def read_observations(mirte):
    time.sleep(cfg.CAMERA_SETTLE_SECONDS)
    return observe_mirte(mirte,allowed_ids=cfg.LANDMARKS)


def scan_localize(localizer,mirte,initialize=False):
    """Gather both IDs without translating, predict every scan rotation.

    Cached bearings are rotated into the scan's final robot frame. They are
    used once for a particle proposal/consistency check, not repeatedly treated
    as independent measurements. A scan has a bounded rotation budget.
    """
    cache={};yaw=0.0;travel=0.0
    for attempt in range(cfg.MAX_SCAN_STEPS+1):
        obs=read_observations(mirte)
        if not initialize:
            localizer.correct(obs);localizer.resample()
        frame_yaw=yaw if initialize else float(localizer.estimate_pose()[2])
        for marker_id,r,b in localizer.valid_observations(obs):
            cache[marker_id]=(r,float(wrap_angle(b+frame_yaw)),frame_yaw)
        if len(cache)>=2:
            combined=[(i,r,float(wrap_angle(b-frame_yaw))) for i,(r,b,_) in cache.items()]
            # Reject incompatible measurements before any motor-goal decision.
            a,b=combined[:2]
            qa=np.array([-a[1]*math.sin(a[2]),a[1]*math.cos(a[2])])
            qb=np.array([-b[1]*math.sin(b[2]),b[1]*math.cos(b[2])])
            baseline=np.linalg.norm(localizer.landmarks[a[0]]-localizer.landmarks[b[0]])
            rotation_separation=abs(cache[a[0]][2]-cache[b[0]][2])
            allowance=3*localizer.sigma_range+baseline*cfg.ALPHA_ROTATION*rotation_separation
            if abs(np.linalg.norm(qb-qa)-baseline)>allowance:
                print('Inconsistent landmark measurements; localization rejected.')
                return None
            if initialize:
                localizer.initialize_from_observations(combined)
            # Correct only with new observations actually taken at this heading.
            for _ in range(cfg.LOCALIZATION_FRAMES):
                current=read_observations(mirte)
                if not current:continue
                localizer.correct(current);localizer.resample()
            residuals=localizer.measurement_residuals(combined)
            range_ok=all(abs(dr)<=3*localizer.sigma_range for _,dr,_ in residuals)
            bearing_ok=all(abs(db)<=3*localizer.sigma_bearing+cfg.ALPHA_ROTATION*rotation_separation for _,_,db in residuals)
            # Include the systematic yaw uncertainty of sequential scan readings.
            scan_position_uncertainty=baseline*localizer.angle_std()/2
            if localizer.confident(cfg.LOCALIZATION_POSITION_STD_M,cfg.LOCALIZATION_ANGLE_STD_RAD) and range_ok and bearing_ok:
                range_midpoint_distance=math.sqrt(max(0.,(a[1]**2+b[1]**2)/2-baseline**2/4))
                return {'pose':localizer.estimate_pose(),'scan_uncertainty':scan_position_uncertainty,'ids':set(cache),'range_midpoint_distance':range_midpoint_distance}
            print('Landmarks found; gathering another view to reduce pose uncertainty.')
            initialize=False
            frame_yaw=float(localizer.estimate_pose()[2])
            cache={i:(r,float(wrap_angle(b+frame_yaw)),frame_yaw) for i,r,b in localizer.valid_observations(current)}
        if attempt==cfg.MAX_SCAN_STEPS:break
        if not initialize and cache:
            missing=[i for i in cfg.LANDMARKS if i not in cache]
            if missing:
                p=localizer.estimate_pose();dx,dz=localizer.landmarks[missing[0]]-p[:2]
                error=float(wrap_angle(math.atan2(-dx,dz)-p[2]))
                angle=float(np.clip(error,-cfg.MAX_ROTATION_STEP_RAD,cfg.MAX_ROTATION_STEP_RAD))
                if abs(angle)<math.radians(2):angle=cfg.MAX_ROTATION_STEP_RAD
            else:angle=cfg.MAX_ROTATION_STEP_RAD
        else:angle=cfg.MAX_ROTATION_STEP_RAD
        if travel+abs(angle)>2*math.pi+1e-6:break
        movement=rotate_step(mirte,angle);localizer.predict(*movement)
        yaw+=movement[1];travel+=abs(movement[1])
    print('Could not observe both configured landmarks in one stationary scan.')
    return None


def stationary_localization(localizer,mirte,frames=None):
    result=scan_localize(localizer,mirte,initialize=True)
    return None if result is None else result['pose']


def main():
    if cfg.LANDMARK_ID_A==cfg.LANDMARK_ID_B or cfg.LANDMARK_DISTANCE_M<=0:
        raise ValueError('Two distinct IDs and a positive measured separation are required.')
    if abs(cfg.STRAIGHT_ANGULAR_BIAS)>1e-9:
        raise ValueError('This controller uses straight translation. Calibrate motors instead of setting STRAIGHT_ANGULAR_BIAS.')
    mirte=KU_Mirte();localizer=MCL(cfg.LANDMARKS);goal=np.asarray(cfg.GOAL,float)
    status=False
    try:
        stop(mirte);time.sleep(1)
        result=scan_localize(localizer,mirte,initialize=True)
        if result is None:
            print('No reliable initial localization. No translation performed.');return False
        since_scan=0.0;scanned_since_translation=True;final_stage_scan=False
        for step in range(cfg.MAX_CONTROL_STEPS):
            pose=localizer.estimate_pose();distance=float(np.linalg.norm(goal-pose[:2]))
            if (distance<=cfg.GOAL_SCAN_DISTANCE_M and not final_stage_scan) or since_scan>=cfg.RESCAN_TRANSLATION_M or not localizer.confident():
                result=scan_localize(localizer,mirte)
                if result is None:return False
                since_scan=0.0;scanned_since_translation=True;final_stage_scan=distance<=cfg.GOAL_SCAN_DISTANCE_M;pose=localizer.estimate_pose();distance=float(np.linalg.norm(goal-pose[:2]))
            if final_stage_scan and since_scan<=cfg.FINAL_MAX_UNOBSERVED_DISTANCE_M:
                bound=distance+2*np.max(localizer.position_std())+result['scan_uncertainty']
                range_consistent=(not scanned_since_translation or result['range_midpoint_distance'] <= cfg.GOAL_TOLERANCE_M + 2*localizer.sigma_range)
                if bound<=cfg.GOAL_TOLERANCE_M and range_consistent:
                    print(f'Goal reached with recent observations of both IDs. Estimated distance {distance:.3f} m, model-based check {bound:.3f} m.')
                    status=True;return True
            dx,dz=goal-pose[:2];error=float(wrap_angle(math.atan2(-dx,dz)-pose[2]))
            print(f'Step {step+1}: pose={pose}, goal distance={distance:.3f}, std={localizer.position_std()}')
            if abs(error)>cfg.HEADING_TOLERANCE_RAD:
                movement=rotate_step(mirte,error)
            else:
                desired=distance if distance>cfg.GOAL_SCAN_DISTANCE_M else max(0.,distance-cfg.FINAL_APPROACH_STANDOFF_M)
                if not final_stage_scan and distance>cfg.GOAL_SCAN_DISTANCE_M:
                    desired=min(desired,max(.005,distance-cfg.GOAL_SCAN_DISTANCE_M))
                if desired<.005:
                    print("Too close for another controlled step, but goal confidence is insufficient.");return False
                movement=translate_step(mirte,min(desired,cfg.MAX_TRANSLATION_STEP_M))
                since_scan+=movement[0];scanned_since_translation=False
            localizer.predict(*movement)
            obs=read_observations(mirte)
            localizer.correct(obs);localizer.resample()
            if not obs and (since_scan>=cfg.RESCAN_TRANSLATION_M or not localizer.confident()):
                result=scan_localize(localizer,mirte)
                if result is None:return False
                since_scan=0.0;scanned_since_translation=True;final_stage_scan=distance<=cfg.GOAL_SCAN_DISTANCE_M
        print('Control budget reached; goal has not been confirmed.');return False
    finally:
        stop(mirte)
        print('Final result:', 'confirmed goal' if status else 'stopped without confirming goal')


if __name__=='__main__':main()
