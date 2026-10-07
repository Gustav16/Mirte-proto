"""Continuous command tracking and geometric arc checks."""
import math
import time
import numpy as np
from mcl import wrap_angle
from robot_io import set_velocity

def integrate_pose(pose,v,w,dt):
    out=np.asarray(pose,float).copy();angle=w*dt
    scale=v*dt*np.sinc(angle/(2*math.pi))
    out[0]-=scale*math.sin(out[2]+angle/2)
    out[1]+=scale*math.cos(out[2]+angle/2)
    out[2]=wrap_angle(out[2]+angle)
    return out

class MotionTracker:
    """Integrate old velocity BEFORE replacing it with a new command."""
    def __init__(self,mirte,localizer=None,pose=None):
        self.mirte=mirte;self.localizer=localizer
        self.pose=np.array([0.,0.,0.]) if pose is None else np.asarray(pose,float).copy()
        self.linear=0.;self.angular=0.;self.timestamp=time.monotonic();self.distance=0.;self.yaw=0.
    def advance(self):
        now=time.monotonic();dt=now-self.timestamp
        if dt<0:raise RuntimeError('Non-monotonic controller clock.')
        self.distance+=abs(self.linear)*dt
        self.yaw+=self.angular*dt
        if self.localizer is not None:
            self.localizer.predict_twist(self.linear,self.angular,dt)
            self.pose=self.localizer.estimate_pose()
        else:self.pose=integrate_pose(self.pose,self.linear,self.angular,dt)
        self.timestamp=now
        return dt
    def command(self,linear,angular):
        self.advance()
        # The old command remains active during driver submission latency.
        set_velocity(self.mirte,linear,angular)
        self.advance()
        self.linear=float(linear);self.angular=float(angular)
    def slew(self,desired_v,desired_w,dt,acceleration=.15,angular_acceleration=.8):
        return (float(np.clip(desired_v,self.linear-acceleration*dt,self.linear+acceleration*dt)),
                float(np.clip(desired_w,self.angular-angular_acceleration*dt,self.angular+angular_acceleration*dt)))

def arc_is_free(world_map,pose,v,w,horizon=1.,radius_margin=.002):
    if world_map is None:return True
    previous=np.asarray(pose,float)[:2]
    for t in np.linspace(0,horizon,21)[1:]:
        next_point=integrate_pose(pose,v,w,t)[:2]
        if not world_map.segment_is_free(previous,next_point):return False
        # Reserve chord-to-arc deviation conservatively for the curved segment.
        if any(world_map.distance_to_box(next_point,i)<=world_map.mirte_radius+world_map.clearance_margin+radius_margin
               for i in world_map.get_visible_box_ids()):return False
        previous=next_point
    return True
