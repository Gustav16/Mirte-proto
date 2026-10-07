"""Exercise 5: continuous slow motion with online MCL and camera-view steering.
No timed translation steps, no periodic stop/start sequence during navigation.
"""
import math
import time
import numpy as np
import ex5_config as cfg
from mcl import MCL,wrap_angle
from aruco_measurements import observe_mirte
from robot_io import KU_Mirte,require_continuous_api,front_clearance,stop
from continuous_control import MotionTracker,integrate_pose


def read_observations(mirte):
    return observe_mirte(mirte,allowed_ids=cfg.LANDMARKS)


def scan_localize(localizer,mirte,initialize=True):
    """One continuous startup rotation, followed by stationary refinement.

    Translation stays zero while the initial position is unknown. This scan
    occurs before driving, not every few centimetres during driving.
    """
    require_continuous_api(mirte);tracker=MotionTracker(mirte,localizer)
    cache={};started=time.monotonic();seeded=not initialize;hold=0;seen=set()
    try:
        while time.monotonic()-started<cfg.SMOOTH_SCAN_MAX_TIME_S:
            tick=time.monotonic();obs=localizer.valid_observations(read_observations(mirte));tracker.advance()
            if seeded:
                localizer.correct(obs);localizer.resample();tracker.pose=localizer.estimate_pose()
            yaw=tracker.yaw if not seeded else float(tracker.pose[2])
            for i,r,b in obs:cache[i]=(r,float(wrap_angle(b+yaw)));seen.add(i)
            if not seeded and len(cache)>=2:
                combined=[(i,r,float(wrap_angle(b-yaw))) for i,(r,b) in cache.items()]
                qa=np.array([-combined[0][1]*math.sin(combined[0][2]),combined[0][1]*math.cos(combined[0][2])])
                qb=np.array([-combined[1][1]*math.sin(combined[1][2]),combined[1][1]*math.cos(combined[1][2])])
                baseline=np.linalg.norm(localizer.landmarks[combined[0][0]]-localizer.landmarks[combined[1][0]])
                if abs(np.linalg.norm(qb-qa)-baseline)>3*localizer.sigma_range+.15:
                    print('Startup landmark geometry inconsistent.');return None
                localizer.initialize_from_observations(combined);seeded=True;tracker.pose=localizer.estimate_pose()
                cache={};tracker.command(0.,0.)
            if seeded:
                if obs:hold+=1
                if hold>=cfg.SMOOTH_SCAN_HOLD_FRAMES and localizer.confident(cfg.LOCALIZATION_POSITION_STD_M,cfg.LOCALIZATION_ANGLE_STD_RAD):
                    tracker.command(0.,0.)
                    return {'pose':localizer.estimate_pose(),'ids':seen}
                if hold<cfg.SMOOTH_SCAN_HOLD_FRAMES:tracker.command(0.,0.)
                else:
                    # Obtain another view if one-landmark refinement is insufficient.
                    tracker.command(0.,cfg.SMOOTH_SCAN_SPEED_RAD_S)
            else:tracker.command(0.,cfg.SMOOTH_SCAN_SPEED_RAD_S)
            if abs(tracker.yaw)>2*math.pi+math.pi/2:return None
            elapsed=time.monotonic()-tick
            if elapsed<cfg.SMOOTH_CONTROL_PERIOD_S:time.sleep(cfg.SMOOTH_CONTROL_PERIOD_S-elapsed)
        print('No reliable startup localization.');return None
    finally:
        # Startup scan ends here once, not between rotation ticks.
        tracker.command(0.,0.)


def stationary_localization(localizer,mirte,frames=None):
    result=scan_localize(localizer,mirte)
    return None if result is None else result['pose']


def landmark_arc_free(localizer,pose,v,w):
    # Conservative circle containing each rectangular box, plus robot geometry.
    box_radius=math.hypot(cfg.BOX_WIDTH_M/2,cfg.BOX_DEPTH_M/2)
    radius=box_radius+cfg.MIRTE_RADIUS_M+cfg.BOX_CLEARANCE_MARGIN_M
    radius+=2*float(np.max(localizer.position_std()))
    return all(np.linalg.norm(integrate_pose(pose,v,w,t)[:2]-point)>radius
               for t in np.linspace(0,cfg.SMOOTH_COLLISION_HORIZON_S,15)
               for i,marker in localizer.landmarks.items()
               for point in [marker+np.asarray(cfg.LANDMARK_BOX_CENTER_OFFSETS_M[i])])


def navigate(localizer,mirte):
    """One uninterrupted velocity loop. View steering occurs while moving.

    At close range the camera alternates between landmarks at low speed, rather
    than stopping every translation step to scan. Final approach is bounded by
    distance since the last observation of EACH landmark and pose uncertainty.
    """
    tracker=MotionTracker(mirte,localizer);goal=np.asarray(cfg.GOAL,float)
    started=time.monotonic();last_seen={i:(started,0.) for i in cfg.LANDMARKS}
    target_id=None;target_since=started;last_any_distance=0.;final_approach=False
    for tick in range(cfg.SMOOTH_MAX_TICKS):
        tick_start=time.monotonic()
        if tick_start-started>cfg.SMOOTH_MAX_TIME_S:
            print('Navigation timeout; goal not confirmed.');return False
        obs=localizer.valid_observations(read_observations(mirte));tracker.advance()
        localizer.correct(obs);localizer.resample();tracker.pose=localizer.estimate_pose()
        pose=tracker.pose;now=time.monotonic()
        for i,r,b in obs:last_seen[i]=(now,tracker.distance)
        if obs:last_any_distance=tracker.distance
        distance=float(np.linalg.norm(goal-pose[:2]))
        oldest_distance=max(tracker.distance-record[1] for record in last_seen.values())
        confidence=localizer.confident(.08,math.radians(15))
        if not confidence:
            print('Pose uncertainty too large; smooth navigation stopped.');return False
        if distance<.19 and oldest_distance<.08:
            final_approach=True
        goal_bound=distance+2*float(np.max(localizer.position_std()))
        if final_approach and goal_bound<=cfg.GOAL_TOLERANCE_M and oldest_distance<=cfg.SMOOTH_MAX_BLIND_DISTANCE_M:
            print(f'Goal reached; model-based distance/uncertainty check {goal_bound:.3f} m.');return True
        if tracker.distance-last_any_distance>cfg.SMOOTH_MAX_BLIND_DISTANCE_M:
            print('Visual measurements absent for too much movement; stopped.');return False
        clearance=front_clearance(mirte,cfg.SONAR_STOP_DISTANCE_M)
        if clearance<=cfg.SONAR_STOP_DISTANCE_M:
            print('Sonar safety stop.');return False
        dx,dz=goal-pose[:2];goal_heading=math.atan2(-dx,dz)
        goal_error=float(wrap_angle(goal_heading-pose[2]))
        # Keep the camera supplied with observations without stationary scans.
        view_needed=(distance<1.35 or not obs or oldest_distance>.12) and not final_approach
        if view_needed:
            if target_id is None:target_id=min(last_seen,key=lambda i:last_seen[i][0]);target_since=now
            if any(i==target_id for i,r,b in obs) and now-target_since>.5:
                target_id=next(i for i in cfg.LANDMARKS if i!=target_id);target_since=now
            if now-target_since>cfg.SMOOTH_VIEW_MAX_TIME_S:
                print('Cannot reacquire a landmark while driving; stopped.');return False
            lx,lz=localizer.landmarks[target_id];heading=math.atan2(-(lx-pose[0]),lz-pose[1])
            error=float(wrap_angle(heading-pose[2]))
            w=float(np.clip(1.5*error,-cfg.SMOOTH_ANGULAR_SPEED_RAD_S,cfg.SMOOTH_ANGULAR_SPEED_RAD_S))
            # Positive, small forward velocity during normal view sweeps.
            v=min(.025,max(cfg.SMOOTH_MIN_VIEW_SPEED_M_S,.04*max(0.,math.cos(goal_error))))
            if distance<.45:v=cfg.SMOOTH_MIN_VIEW_SPEED_M_S
            if abs(goal_error)>math.radians(110):v=0.  # genuine reorientation, not a control-tick stop
        else:
            target_id=None
            w=float(np.clip(1.5*goal_error,-cfg.SMOOTH_ANGULAR_SPEED_RAD_S,cfg.SMOOTH_ANGULAR_SPEED_RAD_S))
            v=min(cfg.SMOOTH_LINEAR_SPEED_M_S,.5*distance)*max(0.,math.cos(goal_error))**2
            if abs(goal_error)>math.radians(75):v=0.
        if clearance<.45:v=min(v,.025)
        v,w=tracker.slew(v,w,cfg.SMOOTH_CONTROL_PERIOD_S,cfg.SMOOTH_ACCELERATION_M_S2,cfg.SMOOTH_ANGULAR_ACCELERATION_RAD_S2)
        if not landmark_arc_free(localizer,pose,v,w):
            print('Predicted curve is too close to a landmark box; stopped.');return False
        tracker.command(v,w)
        if tick%10==0:print(f'pose={pose}, distance={distance:.3f}, v={v:.3f}, w={w:.3f}, std={localizer.position_std()}')
        elapsed=time.monotonic()-tick_start
        if elapsed<cfg.SMOOTH_CONTROL_PERIOD_S:time.sleep(cfg.SMOOTH_CONTROL_PERIOD_S-elapsed)
    print('Navigation tick budget exhausted.');return False


def main():
    if cfg.LANDMARK_ID_A==cfg.LANDMARK_ID_B or cfg.LANDMARK_DISTANCE_M<=0:
        raise ValueError('Two distinct IDs and a positive measured separation required.')
    mirte=KU_Mirte();localizer=MCL(cfg.LANDMARKS);status=False
    try:
        require_continuous_api(mirte)
        result=scan_localize(localizer,mirte)
        if result is None:
            print('No reliable initial localization. No translation performed.');return False
        status=navigate(localizer,mirte)
        return status
    finally:
        stop(mirte)
        print('Final result:', 'confirmed goal' if status else 'stopped without confirming goal')


if __name__=='__main__':main()
