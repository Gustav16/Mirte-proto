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


class VirtualClock:
    def __init__(self,bot):self.now=0.;self.bot=bot
    def monotonic(self):return self.now
    def sleep(self,seconds):
        from continuous_control import integrate_pose
        seconds=max(0.,float(seconds))
        self.bot.pose=integrate_pose(self.bot.pose,self.bot.linear*self.bot.translation_scale,
                                     self.bot.angular*self.bot.rotation_scale,seconds)
        self.now+=seconds


class SimulatedMirte:
    def __init__(self,pose=(.6,-1.8,0),translation_scale=1.,rotation_scale=1.,visible=True):
        self.pose=np.array(pose,float);self.translation_scale=translation_scale;self.rotation_scale=rotation_scale
        self.visible=visible;self.commands=[];self.sonar={'front_left':2.,'front_right':2.}
        self.linear=0.;self.angular=0.;self.stop_calls=0;self.clock=VirtualClock(self)
    def stop(self):self.stop_calls+=1;self.linear=0.;self.angular=0.
    def drive(self,v,w,duration,blocking=True,interrupt=False):
        self.clock.sleep(.02)  # old velocity remains active until submission returns
        self.commands.append((v,w,duration));self.linear=v;self.angular=w
        if duration is not None:
            self.clock.sleep(duration);self.linear=0.;self.angular=0.
    def observations(self):
        self.clock.sleep(.03)  # moving camera readout; observation at receipt time
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
    from unittest.mock import patch
    bot=SimulatedMirte(pose,translation_scale,rotation_scale,visible)
    np.random.seed(seed);output=io.StringIO()
    with patch.object(runner,'KU_Mirte',lambda:bot),patch.object(runner,'observe_mirte',lambda *a,**k:bot.observations()), \
         patch('time.monotonic',bot.clock.monotonic),patch('time.sleep',bot.clock.sleep),contextlib.redirect_stdout(output):
        success=runner.main()
    first=next((i for i,c in enumerate(bot.commands) if c[0]>0),len(bot.commands))
    cruise=bot.commands[first:]
    return {'seed':seed,'reported_success':bool(success),'physical_simulated_goal_error_m':float(np.linalg.norm(bot.pose[:2]-cfg.GOAL)),
            'translations':sum(v!=0 for v,w,t in bot.commands),'rotations':sum(w!=0 for v,w,t in bot.commands),
            'stop_calls':bot.stop_calls,'intermediate_zero_velocity_commands':sum(v==0 and w==0 for v,w,t in cruise),
            'elapsed_simulated_seconds':bot.clock.now,'log':output.getvalue()}


def main():
    stationary=stationary_sweep();ideal=[];drift=[]
    for seed in range(20):
        ideal.append({k:v for k,v in run_controller(seed).items() if k!='log'})
        drift.append({k:v for k,v in run_controller(seed,.9,1.1).items() if k!='log'})
    result={'stationary':stationary,'controller_ideal':ideal,'controller_10percent_drift':drift,
            'assumptions':'Synthetic planar observations with horizontal FOV; continuous velocity replacement with 30ms image reads and 20ms driver submission latency; Front marker faces limited to 75-degree incidence. Occlusion and camera staleness not modeled. Controller observations are noise-free; stationary sweep includes measurement noise.'}
    for records in (ideal,drift):
        result_key='ideal_summary' if records is ideal else 'drift_summary'
        result[result_key]={'confirmed':sum(r['reported_success'] for r in records),
            'false_successes':sum(r['reported_success'] and r['physical_simulated_goal_error_m']>cfg.GOAL_TOLERANCE_M for r in records)}
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':main()
