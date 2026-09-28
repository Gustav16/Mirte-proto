import numpy as np
import particle

class MCL:
    "MCL (monte-carlo localization) class"
    def __init__(self, 
                    prior,
                    alpha1 = 0.05,  # rotation noise from rotation
                    alpha2 = 0.05,  # rotation noise from translation
                    alpha3 = 0.02, # translation noise from translation  
                    alpha4 = 0.02  # translation noise from rotation

                     ):
        self.alpha1 = alpha1,  # rotation noise from rotation
        self.alpha2 = alpha2,  # rotation noise from translation
        self.alpha3 = alpha3, # translation noise from translation  
        self.alpha4 = alpha4  # translation noise from rotation
        self.particles = prior.copy()
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
 
 
    def exponential_pdf(self, z, z_true, lam):
        """Exp(z | lam), truncated/normalized on [0, z_true]. Used for p_short."""
        z = np.asarray(z, dtype=float)
        eta = 1.0 / (1.0 - np.exp(-lam * z_true))  # normalizer so it integrates to 1 on [0, z_true]
        inside = (z >= 0) & (z <= z_true)
        return np.where(inside, eta * lam * np.exp(-lam * z), 0.0)

    def true_distance_to_landmark(self, particles, landmark_xy):
        """
        particles: (N, 3) array of [x, y, theta] poses
        landmark_xy: (2,) array [lx, ly]
        returns: (N,) array of Euclidean distances from each particle to the landmark
        """
        particles = np.asarray(particles, dtype=float)
        dx = landmark_xy[0] - particles.getX()
        dy = landmark_xy[1] - particles.getY()
        return np.sqrt(dx ** 2 + dy ** 2)
 
 
    def observation_model_aruco(self,
        z_measured,
        particles,
        landmark_xy,
        sigma_hit=0.05,   # meters -- tune to your camera's measured noise
        z_min=0.10,       # meters -- closest distance ArUco can be reliably detected
        z_max=3.00,       # meters -- farthest distance ArUco can be reliably detected
        ):
        """
        Weight each particle by how well its implied distance to the landmark
        matches the actual camera measurement z_measured.
    
        z_measured: scalar, the distance reported by the camera/ArUco detector
        particles:  (N, 3) array of [x, y, theta] particle poses
        landmark_xy: (2,) array, known position of the landmark in the map
    
        returns: (N,) array of (unnormalized) weights
        """
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
        theta_new = x_last[2] + u[1]
        x_prime = x_last[0] + d_trans_est*np.cos(theta_new + d_rot_1_est)
        y_prime = x_last[1] + d_trans_est*np.sin(theta_new + d_rot_1_est)
        theta_prime = x_last[2] + d_rot_1_est + d_rot_2_est
        return np.array([x_prime, y_prime, theta_prime])

    def is_state_possible(self, x_t, m):
        return not m.in_collision((x_t.getX(), x_t.getY()))
    
    def sample_motion_model_with_map(self, u, x_last, m):
        "sampling with map"
        q = 0
        while not q:
            x_t = self.sample_motion_model(u,x_last)
            q = self.is_state_possible(x_t, m)
        # if prop then return that
        return x_t

    def measurement_model(self ,z, x, m):
        return  self.observation_model_aruco()

    def mcl(self, u, z, m):
        "MCL function"
        particles = self.sample_motion_model_with_map(u, self.particles, m)
        particles.setWeight = self.measurement_model(z, particles, m)
        particles.setWeight = np.cumsum(particles.getWeight() /np.sum(particles.getWeight())) #smooth weights

        #redraw samples
        picks = np.random.rand(self.M)
        indices = np.searchsorted(particles.getWeight, picks) #use binsearch to get indices

        #set new belief distribution
        self.particles = particles[indices]
        return

    def estimate_pose(self):
        "return estimate of pose as everage of particles"
        return particle.Particle.estimate_pose(self.particles)
    