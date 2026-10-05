import numpy as np
import random_numbers as rn


class Particle(object):
    """Data structure for storing particle information (state and weight)"""
    def __init__(self, x=0.0, y=0.0, theta=0.0, weight=0.0):
        self.x = x
        self.y = y
        self.theta = np.mod(theta, 2.0*np.pi)
        self.weight = weight

    def getX(self):
        return self.x
        
    def getY(self):
        return self.y
        
    def getTheta(self):
        return self.theta
        
    def getWeight(self):
        return self.weight

    def setX(self, val):
        self.x = val

    def setY(self, val):
        self.y = val

    def setTheta(self, val):
        self.theta = np.mod(val, 2.0*np.pi)

    def setWeight(self, val):
        self.weight = val


def estimate_pose(particles_list, weighted=False):
    """Estimate the pose from particles by computing the average position and orientation over all particles. 
    Set weighted=True to use stored weights; the default assumes a resampled population."""
    if not particles_list: return Particle()
    weights=np.array([p.weight for p in particles_list],float) if weighted else np.ones(len(particles_list))
    if np.any(weights<0) or not np.all(np.isfinite(weights)) or weights.sum()<=0: raise ValueError("Invalid particle weights")
    weights/=weights.sum()
    x=sum(w*p.x for w,p in zip(weights,particles_list));y=sum(w*p.y for w,p in zip(weights,particles_list))
    theta=np.arctan2(sum(w*np.sin(p.theta) for w,p in zip(weights,particles_list)),sum(w*np.cos(p.theta) for w,p in zip(weights,particles_list)))
    return Particle(x,y,theta)


def move_particle(particle, delta_x, delta_y, delta_theta):
    """Move the particle by (delta_x, delta_y, delta_theta)."""
    particle.setX(particle.getX() + delta_x)
    particle.setY(particle.getY() + delta_y)
    particle.setTheta(particle.getTheta() + delta_theta)
    return particle


def add_uncertainty(particles_list, sigma, sigma_theta):
    """Add some noise to each particle in the list. Sigma and sigma_theta is the noise
    standard deviations for position and angle noise."""
    for particle in particles_list:
        particle.x += rn.randn(0.0, sigma)
        particle.y += rn.randn(0.0, sigma)
        particle.theta = np.mod(particle.theta + rn.randn(0.0, sigma_theta), 2.0 * np.pi) 


def add_uncertainty_von_mises(particles_list, sigma, theta_kappa):
    """Add some noise to each particle in the list. Sigma is the position standard deviation; theta_kappa is angular concentration."""
    for particle in particles_list:
        particle.x += rn.randn(0.0, sigma)
        particle.y += rn.randn(0.0, sigma)
        particle.theta = np.mod(rn.rand_von_mises(particle.theta, theta_kappa) + np.pi, 2.0 * np.pi) - np.pi
