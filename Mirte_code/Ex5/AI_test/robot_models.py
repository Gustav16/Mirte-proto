"""Planner dynamics. PointMassModel uses x/z; MirteModel uses x/z/theta."""
import math
import numpy as np

class RobotModel:
    def __init__(self,ctrl_range):self.ctrl_range=ctrl_range
    def forward_dyn(self,x,u,T):raise NotImplementedError
    def inverse_dyn(self,x,x_goal,T):raise NotImplementedError

class PointMassModel(RobotModel):
    def forward_dyn(self,x,u,T):
        if T<0 or len(u)<T:raise ValueError('Invalid integration horizon.')
        p=np.asarray(x,float).copy();out=[]
        for command in u[:T]:p=p+command;out.append(p.copy())
        return out
    def inverse_dyn(self,x,x_goal,T):
        x,x_goal=np.asarray(x,float),np.asarray(x_goal,float)
        if T<1:raise ValueError('T must be positive.')
        dist=float(np.linalg.norm(x_goal-x))
        if dist<=1e-12:return [x.copy() for _ in range(T)]
        step=min(float(self.ctrl_range[1]),dist/T)
        if step<=0:raise ValueError('Positive maximum displacement required.')
        u=np.tile((x_goal-x)/dist*step,(T,1));return self.forward_dyn(x,u,T)

class MirteModel(RobotModel):
    """Stateless model: orientation belongs to each state/branch, never the model."""
    def forward_dyn(self,x,u,T):
        x=np.asarray(x,float).copy()
        if x.shape!=(3,):raise ValueError('MirteModel state must be [x,z,theta].')
        out=[]
        for ds,da in u[:T]:
            x[2]=math.atan2(math.sin(x[2]+da),math.cos(x[2]+da))
            x[0]-=math.sin(x[2])*ds;x[1]+=math.cos(x[2])*ds;out.append(x.copy())
        return out
    def inverse_dyn(self,x,x_goal,T):
        x=np.asarray(x,float);goal=np.asarray(x_goal,float)
        if x.shape!=(3,) or T<1:raise ValueError('3D pose and positive T required.')
        dx,dz=goal[:2]-x[:2];d=float(np.hypot(dx,dz))
        if d<1e-12:return [x.copy()]
        da=math.atan2(-dx,dz)-x[2];da=math.atan2(math.sin(da),math.cos(da))
        commands=[]
        if abs(da)>1e-8:commands.append((0.,da))
        while len(commands)<T and d>1e-12:
            ds=min(self.ctrl_range[1],d);commands.append((ds,0.));d-=ds
        return self.forward_dyn(x,commands,len(commands))
