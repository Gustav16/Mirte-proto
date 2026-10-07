"""Weighted Monte Carlo localisation in a FIXED landmark frame.
x right, z forward, theta/bearing positive left; units metres and radians.
"""
import math
import numpy as np
import ex5_config as cfg


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class MCL:
    def __init__(self, landmarks, number_of_particles=cfg.NUMBER_OF_PARTICLES,
                 x_bounds=(cfg.INITIAL_X_MIN,cfg.INITIAL_X_MAX),
                 z_bounds=(cfg.INITIAL_Z_MIN,cfg.INITIAL_Z_MAX),
                 sigma_range=cfg.SIGMA_RANGE_M, sigma_bearing=cfg.SIGMA_BEARING_RAD,
                 alpha_translation=cfg.ALPHA_TRANSLATION, alpha_rotation=cfg.ALPHA_ROTATION,
                 initial_pose=None, initial_std=(0.02,0.02,math.radians(2)),
                 augmented=None, fast_const=cfg.AUGMENTED_MCL_FAST_ALPHA,
                 slow_const=cfg.AUGMENTED_MCL_SLOW_ALPHA,
                 max_random_fraction=cfg.AUGMENTED_MCL_MAX_RANDOM_FRACTION,
                 recovery_pose_is_free=None):
        if not hasattr(landmarks,'items'):
            raise TypeError('MCL needs {id: marker_position_in_fixed_frame}, not a LocalMap.')
        self.landmarks={int(i):np.asarray(p,float).copy() for i,p in landmarks.items()}
        if any(p.shape != (2,) or not np.all(np.isfinite(p)) for p in self.landmarks.values()):
            raise ValueError('Landmark coordinates must be finite 2D positions.')
        self.number_of_particles=int(number_of_particles)
        self.sigma_range=float(sigma_range);self.sigma_bearing=float(sigma_bearing)
        self.alpha_translation=float(alpha_translation);self.alpha_rotation=float(alpha_rotation)
        self.x_bounds=tuple(x_bounds);self.z_bounds=tuple(z_bounds)
        if self.number_of_particles<2 or min(self.sigma_range,self.sigma_bearing)<=0:
            raise ValueError('At least two particles and positive measurement sigmas required.')
        if not np.all(np.isfinite([self.sigma_range,self.sigma_bearing,self.alpha_translation,self.alpha_rotation])):
            raise ValueError("Noise parameters must be finite.")
        if min(self.alpha_translation,self.alpha_rotation)<0:
            raise ValueError('Negative motion noise is invalid.')
        if any(len(b)!=2 or not np.all(np.isfinite(b)) or b[0]>=b[1] for b in (self.x_bounds,self.z_bounds)):
            raise ValueError('Invalid prior bounds.')
        self.weights=np.full(self.number_of_particles,1/self.number_of_particles)
        self.particles=np.column_stack([
            np.random.uniform(*self.x_bounds,self.number_of_particles),
            np.random.uniform(*self.z_bounds,self.number_of_particles),
            np.random.uniform(-math.pi,math.pi,self.number_of_particles)])
        self.has_measurements=False
        self.augmented=cfg.AUGMENTED_MCL_ENABLED if augmented is None else bool(augmented)
        self.fast_const=float(fast_const);self.slow_const=float(slow_const)
        self.max_random_fraction=float(max_random_fraction)
        if not 0<self.slow_const<self.fast_const<1 or not 0<=self.max_random_fraction<=1:
            raise ValueError('Require 0 < slow_const < fast_const < 1 and fraction in [0,1].')
        self.recovery_pose_is_free=recovery_pose_is_free
        self._log_fast=None;self._log_slow=None;self._recovery_ids=None
        self.random_fraction=0.;self.last_random_particles=0
        if initial_pose is not None:self.initialize_pose(initial_pose,initial_std)

    def reset_recovery(self):
        self._log_fast=None;self._log_slow=None;self._recovery_ids=None
        self.random_fraction=0.;self.last_random_particles=0

    def _track_measurement_quality(self,log_likelihood,ids):
        """Colleague's fast/slow averages, in log-space to avoid underflow.

        Compare like-for-like landmark sets; a missing ID is not evidence of
        teleportation. Kernel constants cancel for an unchanged ID set.
        """
        if not self.augmented:return
        evidence=log_likelihood+np.log(np.maximum(self.weights,np.finfo(float).tiny))
        top=float(np.max(evidence))
        avg=top+math.log(float(np.exp(evidence-top).sum()))
        if self._recovery_ids!=ids or self._log_fast is None:
            self._log_fast=self._log_slow=avg;self._recovery_ids=ids
            self.random_fraction=0.;return
        def smooth(old,alpha):
            return float(np.logaddexp(math.log1p(-alpha)+old,math.log(alpha)+avg))
        self._log_fast=smooth(self._log_fast,self.fast_const)
        self._log_slow=smooth(self._log_slow,self.slow_const)
        difference=self._log_fast-self._log_slow
        probability=0. if difference>=0 else -math.expm1(difference)
        self.random_fraction=min(self.max_random_fraction,probability)

    def _random_recovery_poses(self,count):
        """Bounded map-constrained sampling; never wait forever on a full map."""
        accepted=[];attempts=0;limit=max(100,count*50)
        while len(accepted)<count and attempts<limit:
            pose=np.array([np.random.uniform(*self.x_bounds),np.random.uniform(*self.z_bounds),
                           np.random.uniform(-math.pi,math.pi)])
            attempts+=1
            if self.recovery_pose_is_free is None or self.recovery_pose_is_free(pose):accepted.append(pose)
        if len(accepted)<count:raise RuntimeError('No free recovery poses within sampling budget.')
        return np.asarray(accepted).reshape(count,3)

    def initialize_pose(self,pose,std=(0.02,0.02,math.radians(2))):
        pose=np.asarray(pose,float);std=np.asarray(std,float)
        if pose.shape!=(3,) or std.shape!=(3,) or not np.all(np.isfinite(pose)) or not np.all(np.isfinite(std)) or np.any(std<0):
            raise ValueError('Invalid initial pose/standard deviation.')
        self.particles=pose+np.random.normal(size=(self.number_of_particles,3))*std
        self.particles[:,2]=wrap_angle(self.particles[:,2]);self.weights[:]=1/self.number_of_particles
        self.has_measurements=False;self.reset_recovery()

    def valid_observations(self,observations):
        result=[];seen=set()
        for item in observations:
            if len(item)!=3:continue
            i,r,b=item;i=int(i);r=float(r);b=float(b)
            if i in self.landmarks and i not in seen and r>0 and np.isfinite(r) and np.isfinite(b):
                result.append((i,r,float(wrap_angle(b))));seen.add(i)
        return result

    def initialize_from_observations(self,observations):
        """Sensor-informed particle proposal from TWO distinct landmarks.

        Samples noisy measurements, fits a 2D rigid pose for each sample and
        scores particles with the normal measurement likelihood. This avoids
        losing the correct pose in a sparse uniform prior. All observations
        must refer to the same robot frame; never combine moved-frame data.
        """
        obs=self.valid_observations(observations)
        if len(obs)<2:return False
        a,b=obs[:2];la,lb=self.landmarks[a[0]],self.landmarks[b[0]]
        baseline=lb-la
        if np.linalg.norm(baseline)<1e-6:return False
        n=self.number_of_particles
        q=[]
        for _,r,bearing in (a,b):
            rr=np.maximum(1e-4,r+np.random.normal(0,self.sigma_range,n))
            bb=bearing+np.random.normal(0,self.sigma_bearing,n)
            q.append(np.column_stack([-rr*np.sin(bb),rr*np.cos(bb)]))
        delta=q[1]-q[0]
        theta=wrap_angle(math.atan2(baseline[1],baseline[0])-np.arctan2(delta[:,1],delta[:,0]))
        c,s=np.cos(theta),np.sin(theta)
        qa=np.column_stack([c*q[0][:,0]-s*q[0][:,1],s*q[0][:,0]+c*q[0][:,1]])
        qb=np.column_stack([c*q[1][:,0]-s*q[1][:,1],s*q[1][:,0]+c*q[1][:,1]])
        self.particles[:,:2]=((la-qa)+(lb-qb))/2
        self.particles[:,2]=theta;self.weights[:]=1/n
        # Proposal already incorporates these data: no second likelihood multiplication.
        self.has_measurements=True
        self.reset_recovery()
        return True

    def predict(self,translation_m,rotation_rad):
        ds,da=float(translation_m),float(rotation_rad)
        if not np.isfinite(ds) or not np.isfinite(da):raise ValueError('Invalid motion.')
        if ds==0 and da==0:return
        n=self.number_of_particles
        rs=self.alpha_rotation*abs(da)+cfg.ROTATION_NOISE_FLOOR_RAD
        ts=self.alpha_translation*abs(ds)+(cfg.TRANSLATION_NOISE_FLOOR_M if ds else 0)
        theta=wrap_angle(self.particles[:,2]+da+np.random.normal(0,rs,n))
        trans=ds+np.random.normal(0,ts,n)
        self.particles[:,2]=theta;self.particles[:,0]-=np.sin(theta)*trans;self.particles[:,1]+=np.cos(theta)*trans

    def predict_twist(self,linear,angular,dt):
        """Integrate the previously active v/w over elapsed monotonic time.

        Exact circular-arc integration, with motion noise scaled by sqrt(dt).
        No artificial minimum duration or rotate-then-translate approximation.
        """
        v,w,dt=map(float,(linear,angular,dt))
        if not np.all(np.isfinite([v,w,dt])) or dt<0:raise ValueError('Invalid twist/time interval.')
        if dt==0 or (v==0 and w==0):return
        n=self.number_of_particles
        root=math.sqrt(dt)
        dv=np.random.normal(0,(self.alpha_translation*abs(v)+cfg.TRANSLATION_NOISE_FLOOR_M)*root,n)
        da=w*dt+np.random.normal(0,(self.alpha_rotation*abs(w)+cfg.ROTATION_NOISE_FLOOR_RAD)*root,n)
        ds=v*dt+dv
        theta=self.particles[:,2].copy()
        # sinc form remains stable as the turn angle tends to zero.
        scale=ds*np.sinc(da/(2*math.pi));middle=theta+da/2
        self.particles[:,0]-=scale*np.sin(middle)
        self.particles[:,1]+=scale*np.cos(middle)
        self.particles[:,2]=wrap_angle(theta+da)

    def _log_measurement_weights(self,observations):
        out=np.zeros(self.number_of_particles)
        for i,r,b in self.valid_observations(observations):
            dx=self.landmarks[i][0]-self.particles[:,0];dz=self.landmarks[i][1]-self.particles[:,1]
            er=r-np.hypot(dx,dz)
            eb=wrap_angle(b-wrap_angle(np.arctan2(-dx,dz)-self.particles[:,2]))
            out-=.5*(er/self.sigma_range)**2+.5*(eb/self.sigma_bearing)**2
        return out

    def correct(self,observations):
        obs=self.valid_observations(observations)
        if not obs:return self.estimate_pose()
        likelihood=self._log_measurement_weights(obs)
        self._track_measurement_quality(likelihood,tuple(sorted(i for i,r,b in obs)))
        logw=np.log(np.maximum(self.weights,np.finfo(float).tiny))+likelihood
        logw-=np.max(logw);w=np.exp(logw);total=w.sum()
        if not np.isfinite(total) or total<=0:raise RuntimeError('Invalid particle likelihoods.')
        self.weights=w/total;self.has_measurements=True
        return self.estimate_pose()

    def effective_sample_size(self):return float(1/np.sum(self.weights**2))

    def resample(self,force=False):
        self.last_random_particles=0
        if not force and self.random_fraction<=0 and self.effective_sample_size()>.55*self.number_of_particles:return False
        n=self.number_of_particles;cdf=np.cumsum(self.weights);cdf[-1]=1
        idx=np.searchsorted(cdf,(np.arange(n)+np.random.random())/n)
        self.particles=self.particles[idx].copy()
        self.particles[:,:2]+=np.random.normal(0,cfg.ROUGHEN_POSITION_M,(n,2))
        self.particles[:,2]=wrap_angle(self.particles[:,2]+np.random.normal(0,cfg.ROUGHEN_ANGLE_RAD,n))
        if self.augmented and self.random_fraction>0:
            mask=np.random.random(n)<self.random_fraction
            count=int(mask.sum())
            if count:
                self.particles[mask]=self._random_recovery_poses(count)
                self.reset_recovery();self.last_random_particles=count
        self.weights[:]=1/n
        return True

    def estimate_pose(self,weighted=True):
        w=self.weights if weighted else np.full(self.number_of_particles,1/self.number_of_particles)
        p=np.sum(w[:,None]*self.particles[:,:2],axis=0)
        theta=math.atan2(np.sum(w*np.sin(self.particles[:,2])),np.sum(w*np.cos(self.particles[:,2])))
        return np.array([p[0],p[1],theta])

    def position_std(self):
        p=self.estimate_pose();return np.sqrt(np.sum(self.weights[:,None]*(self.particles[:,:2]-p[:2])**2,axis=0))

    def angle_std(self):
        error=wrap_angle(self.particles[:,2]-self.estimate_pose()[2]);return float(np.sqrt(np.sum(self.weights*error**2)))

    def confident(self,position_limit=0.06,angle_limit=math.radians(12)):
        return self.has_measurements and self.random_fraction<=0 and self.last_random_particles==0 and np.max(self.position_std())<=position_limit and self.angle_std()<=angle_limit

    def measurement_residuals(self,observations):
        p=self.estimate_pose();out=[]
        for i,r,b in self.valid_observations(observations):
            dx,dz=self.landmarks[i]-p[:2]
            out.append((i,r-math.hypot(dx,dz),float(wrap_angle(b-(math.atan2(-dx,dz)-p[2])))))
        return out

    def update(self,movement,observations):
        self.predict(*movement);self.correct(observations);self.resample()
        return self.estimate_pose()
