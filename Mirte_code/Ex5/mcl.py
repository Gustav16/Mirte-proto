import numpy as np
import particle as pcl
import random_numbers as rn

class MCL:
    "MCL (monte-carlo localization) class"
    def __init__(self, 
                    prior,
                    alpha1 = 0.05,  # rotation noise from rotation TODO
                    alpha2 = 0.05,  # rotation noise from translation TODO
                    alpha3 = 0.02, # translation noise from translation  TODO
                    alpha4 = 0.02,  # translation noise from rotation TODO
                    fast_const = 0.9, #we may change this value TODO
                    slow_const = 0.8 #we may change this value TODO

                     ):
        self.alpha1 = alpha1,  # rotation noise from rotation
        self.alpha2 = alpha2,  # rotation noise from translation
        self.alpha3 = alpha3, # translation noise from translation  
        self.alpha4 = alpha4  # translation noise from rotation 
        self.W_fast = None
        self.W_slow = None
        self.fast_const = fast_const
        self.slow_const =  slow_const
        self.particles = prior
        self.M = len(self.particles)

    
    def gaussian_pdf(self, z, mean, std):
        """N(z | mean, std^2), vectorized. std can be scalar or array."""
        std = np.asarray(std, dtype=float)
        return (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(
            -0.5 * ((z - mean) / std) ** 2
        )
 
 
    def uniform_pdf(self, z, low, high):
        """U(z | low, high), vectorized. Zero outside [low, high]."""
        z = np.asarray(self, z, dtype=float)
        width = high - low
        inside = (z >= low) & (z <= high)
        return np.where(inside, 1.0 / width, 0.0)
 

    def true_distance_to_landmark(self, particles, landmark_xy):
        """
        particles: (N, 3) array of [x, y, theta] poses
        landmark_xy: (2,) array [lx, ly]
        returns: (N,) array of Euclidean distances from each particle to the landmark
        """
        delta = np.array([
        landmark_xy[0] - particles.getX(),
        landmark_xy[1] - particles.getY()])

        return np.linalg.norm(delta)
 
 
    def observation_model_aruco(self,
        z_measured,
        particles,
        landmark_xy,
        sigma_hit=0.00483,   # meters
        z_min=0.10,       # meters -- closest distance ArUco can be reliably detected TODO
        z_max=3.00,       # meters -- farthest distance ArUco can be reliably detected TODO
        ):
        """
        Weight each particle by how well its implied distance to the landmark
        matches the actual camera measurement z_measured.
    
        z_measured: scalar, the distance reported by the camera/ArUco detector
        particles:  (N, 3) array of [x, y, theta] particle poses
        landmark_xy: (2,) array, known position of the landmark in the map
    
        returns: (N,) array of (unnormalized) weights
        """
        # TODO: rewrite
        z_true = self.true_distance_to_landmark(particles, landmark_xy)
    
        p_range = self.uniform_pdf(z_true, z_min, z_max)          # gate: is this pose even plausible?
        p_hit = self.gaussian_pdf(z_measured, z_true, sigma_hit)  # how close is measurement to that pose's true distance?
    
        weights = p_range * p_hit
        return weights

    def sample(self, b_variance):
        "function for sampling noise using gaussian"
        return np.random.normal(loc=0, scale=np.sqrt(b_variance)) #zero mean

    def sample_motion_model(self, u, x_last):
        "function for sampling motion model"
        d_rot_1 = u[1]
        d_trans = u[0]
        d_rot_2 = 0 #we dont rotate after first rotation and transportation, but there may be noise

        d_rot_1_est = d_rot_1 + self.sample(self.alpha1*(d_rot_1**2)+ self.alpha2*(d_trans**2))
        d_trans_est = d_trans + self.sample(self.alpha3*(d_trans**2) + self.alpha4*(d_rot_1**2))
        d_rot_2_est = d_rot_2 + self.sample(self.alpha2*(d_trans**2))

        theta_new = x_last.getTheta() + u[1]
        x_prime = x_last.getX() + d_trans_est*np.cos(theta_new + d_rot_1_est)
        y_prime = x_last.getY() + d_trans_est*np.sin(theta_new + d_rot_1_est)
        theta_prime = np.mod(x_last.getTheta() + d_rot_1_est + d_rot_2_est, 2.0 * np.pi) 
        
        return pcl.Particle(x_prime, y_prime, theta_prime)

    def is_state_possible(self, x_t, m):
        return not m.in_collision((x_t.getX(), x_t.getY()))
    
    def sample_motion_model_with_map(self, u, x_last, m):
        "sampling with map"
        q = 0
        while not q:
            x_t = self.sample_motion_model(u,x_last)
            q = self.is_state_possible(x_t, m)
        return x_t

    def measurement_model(self ,z, x, m):
        "measurement model"
        return self.observation_model_aruco(z, x,m)
    
    def add_noise(self, m):
        "function for adding random pose to map"
        while True:
            x = np.random.uniform(m.map_area[0][0], m.map_area[1][0])
            y = np.random.uniform(m.map_area[0][1], m.map_area[1][1])
            theta = np.random.uniform(0, 2*np.pi)

            if not m.in_collision([x,y]):
                return pcl.Particle(x, y, theta, 1.0/self.M)

    def mcl(self, u, z, m):
        "augmented MCL function"
        particles = self.sample_motion_model_with_map(u, self.particles, m)
        weights = self.measurement_model(z, particles, m)
        #keep this for now may not use
        for p, w in zip(particles, weights):
            p.setWeight(w)
        
        w_avg = np.mean(weights)
        self.W_fast += self.fast_const(w_avg - self.W_fast)
        self.W_slow += self.slow_const(w_avg - self.W_slow)
        weights = np.cumsum(weights /np.sum(weights)) #smooth weights
        
        #redraw sample
        random_noise = np.random.rand(self.M) < max(0, 1- self.W_fast/self.W_slow)
        picks = np.random.rand(self.M)
        indices = np.searchsorted(weights, picks)

        new_particles = []
        for i in range(self.M):
            if random_noise[i]:
                new_particles.append(self.add_noise(m))
            else:
                new_particles.append(particles[indices[i]])

        #set new belief distribution
        self.particles = np.array(new_particles)
        return
    
    def estimate_pose(self, particles = None):
        "return estimate of pose as average of particles, or given particle list"
        if particles is None:
            particles = self.particles
        return pcl.Particle.estimate_pose(particles)
    