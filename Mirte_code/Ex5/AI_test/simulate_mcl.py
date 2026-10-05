"""Reproducible stationary filter and controller simulations; no robot needed."""
import contextlib
import io
import math
import json
import numpy as np
import ex5_config as cfg
from mcl import MCL,wrap_angle


def observations_for_pose(pose,noisy=True):
    x,z,theta=pose;out=[]
    for i,(lx,lz) in cfg.LANDMARKS.items():
        dx,dz=lx-x,lz-z
        r=math.hypot(dx,dz);b=float(wrap_angle(math.atan2(-dx,dz)-theta))
        if noisy:r+=np.random.normal(0,.02);b+=np.random.normal(0,math.radians(1.5))
        out.append((i,r,b))
    return out


def stationary_sweep(seeds=100):
    truth=np.array([.35,-.85,math.radians(18)]);errors=[]
    for seed in range(seeds):
        np.random.seed(seed);f=MCL(cfg.LANDMARKS)
        f.initialize_from_observations(observations_for_pose(truth))
        for _ in range(8):f.correct(observations_for_pose(truth));f.resample()
        errors.append(float(np.linalg.norm(f.estimate_pose()[:2]-truth[:2])))
    return {'seeds':seeds,'median_error_m':float(np.median(errors)),'max_error_m':max(errors),'over_10cm':sum(e>.1 for e in errors)}


class SimulatedMirte:
    def __init__(self,pose=(.6,-1.8,0),translation_scale=1.,rotation_scale=1.,visible=True):
        self.pose=np.array(pose,float);self.translation_scale=translation_scale;self.rotation_scale=rotation_scale
        self.visible=visible;self.commands=[];self.sonar={'front_left':2.,'front_right':2.}
    def stop(self):pass
    def drive(self,v,w,duration,blocking=True):
        self.commands.append((v,w,duration))
        self.pose[2]=wrap_angle(self.pose[2]+w*duration*self.rotation_scale)
        self.pose[0]-=math.sin(self.pose[2])*v*duration*self.translation_scale
        self.pose[1]+=math.cos(self.pose[2])*v*duration*self.translation_scale
    def observations(self):
        if not self.visible:return []
        out=[];x,z,theta=self.pose;half=math.atan(cfg.IMAGE_WIDTH/2/cfg.FOCAL_LENGTH_PX)
        for i,r,b in observations_for_pose(self.pose,noisy=False):
            lx,lz=cfg.LANDMARKS[i];dx,dz=lx-x,lz-z
            qx=math.cos(theta)*dx+math.sin(theta)*dz
            qz=-math.sin(theta)*dx+math.cos(theta)*dz-cfg.CAMERA_FORWARD_OFFSET_M
            if dz>0 and abs(math.atan2(dx,dz))<math.radians(75) and qz>0 and abs(math.atan2(qx,qz))<half:out.append((i,r,b))
        return out


def run_controller(seed=7,translation_scale=1.,rotation_scale=1.,visible=True,pose=(.6,-1.8,0)):
    import run_between_boxes as runner
    bot=SimulatedMirte(pose,translation_scale,rotation_scale,visible)
    old=(runner.KU_Mirte,runner.observe_mirte,runner.time.sleep)
    runner.KU_Mirte=lambda:bot;runner.observe_mirte=lambda *a,**k:bot.observations();runner.time.sleep=lambda *a:None
    np.random.seed(seed);output=io.StringIO()
    try:
        with contextlib.redirect_stdout(output):success=runner.main()
    finally:runner.KU_Mirte,runner.observe_mirte,runner.time.sleep=old
    return {'seed':seed,'reported_success':bool(success),'physical_simulated_goal_error_m':float(np.linalg.norm(bot.pose[:2]-cfg.GOAL)),
            'translations':sum(v!=0 for v,w,t in bot.commands),'rotations':sum(w!=0 for v,w,t in bot.commands),'log':output.getvalue()}


def main():
    stationary=stationary_sweep();ideal=[];drift=[]
    for seed in range(20):
        ideal.append({k:v for k,v in run_controller(seed).items() if k!='log'})
        drift.append({k:v for k,v in run_controller(seed,.9,1.1).items() if k!='log'})
    result={'stationary':stationary,'controller_ideal':ideal,'controller_10percent_drift':drift,
            'assumptions':'Synthetic planar observations with horizontal FOV; instantaneous blocking moves; Front marker faces limited to 75-degree incidence. Occlusion and camera staleness not modeled.'}
    for records in (ideal,drift):
        result_key='ideal_summary' if records is ideal else 'drift_summary'
        result[result_key]={'confirmed':sum(r['reported_success'] for r in records),
            'false_successes':sum(r['reported_success'] and r['physical_simulated_goal_error_m']>cfg.GOAL_TOLERANCE_M for r in records)}
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':main()
