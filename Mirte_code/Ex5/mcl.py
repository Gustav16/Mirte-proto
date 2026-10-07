import numpy as np
import particle as pcl
import random_numbers as rn

class MCL:
    "MCL (monte-carlo localization) class"
    def __init__(self, 
                    prior,
                    alpha1 = 0.05,  # rotation noise from rotation TODO
                    alpha2 = 0.05,  # rotation noise from translation TODO
                    alpha3 = 0.05, # translation noise from translation  TODO
                    alpha4 = 0.02,  # translation noise from rotation TODO
                    fast_const = 0.1, #we may change this value TODO
                    slow_const = 0.001 #we may change this value TODO

                     ):
        self.alpha1 = alpha1  # rotation noise from rotation
        self.alpha2 = alpha2  # rotation noise from translation
        self.alpha3 = alpha3 # translation noise from translation  
        self.alpha4 = alpha4  # translation noise from rotation 
        self.W_fast = None
        self.W_slow = None
        self.fast_const = fast_const
        self.slow_const =  slow_const
        self.particles = np.array(prior)
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

    def is_visible(self, bearing, z_max):
        "function that determines if a particle could potentially see a landmark"
        #  TODO decide FOV right now just use 90 degrees
        if abs(bearing) > z_max:
            return False
        return True
 

    def true_z_for_landmarks(self, x_t, m):
        """
        get distance and bearing to landmarks in a dictionary
        """
        res = {}
        particle_x =  x_t.getX()
        particle_y =  x_t.getY()
        particle_theta = x_t.getTheta()
        for pos, id in m.landmarks:
            landmark_x,landmark_y = pos
            delta = np.array([
            landmark_x - particle_x,
            landmark_y - particle_y])
            bearing = np.mod(np.arctan2(delta[1], delta[0]) - particle_theta + np.pi, 
                             2 * np.pi) - np.pi
            res[id] = np.array([np.linalg.norm(delta), bearing]) #get distance and bearing to landmarks in a dictionary
        return res

    def p_hit(self,z_measured ,z_true, sigma_hit, z_max):
        "hit function we assume independece and use the product"
        prop = 1
        for id in z_true:
            #is_visvible_from_pose = self.is_visible(z_true[id][1], z_max[1])
            is_visible_from_pose = True
            measured = z_measured.get(id)

            if measured is None:
                if is_visible_from_pose:
                    prop *= 0.3  # missed detection
                else:
                    continue
            else:
                prop *= self.gaussian_pdf(measured[0], z_true[id][0], sigma_hit[0])
                bearing_error = np.mod(measured[1] - z_true[id][1] + np.pi, 2 * np.pi) - np.pi #wrap to [-pi, pi)
                prop *= self.gaussian_pdf(bearing_error, 0, sigma_hit[1])
        return prop

    def p_range(self, z_measured, z_min, z_max, sigma_hit):
        prop = 1
        for id in z_measured:
            if z_measured[id] is None:
                continue
            measured_dist, measured_bearing = z_measured[id]
            prop*= self.uniform_pdf(measured_dist, z_min[0], z_max[0])
            abs_bearing =  abs(measured_bearing) 
            if abs_bearing >  z_max[1]:
                prop*= self.uniform_pdf(z_min[1], z_min[1], z_max[1])*self.gaussian_pdf(abs_bearing - z_max[1], 0, sigma_hit[1])
            else:
                prop*= self.uniform_pdf(measured_bearing, z_min[1], z_max[1])
        return prop
 
    def observation_model_aruco(self,
        z_measured,
        x_t,
        m,
        sigma_hit=[10, 0.1],   # std deviation of distance (cm) and bearing (rad, ~6 deg) TODO tune on robot
        z_min=[30,-0.34877],       # meters -- closest distance ArUco can be reliably detected
        z_max=[500, 0.34877],       # meters -- farthest distance ArUco can be reliably detected TODO
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
        z_true = self.true_z_for_landmarks(x_t, m)
    
        #p_range = self.p_range(z_measured, z_min, z_max)          # gate: is this pose even plausible?
        p_hit = self.p_hit(z_measured, z_true, sigma_hit, z_max)  # how close is measurement to that pose's true distance?

        #weights = p_range * p_hit
        weights = p_hit
        return weights
    
    def measurement_model(self, z, x_t, m):
        "measurement model"
        return self.observation_model_aruco(z, x_t, m)

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
    
    def add_noise(self, m):
        "function for adding random pose to map"
        while True:
            x = np.random.uniform(m.map_area[0][0], m.map_area[1][0])
            y = np.random.uniform(m.map_area[0][1], m.map_area[1][1])
            theta = np.random.uniform(0, 2*np.pi)

            if not m.in_collision([x,y]):
                return pcl.Particle(x, y, theta, 1.0/self.M)

    def copy_particle(self, particle):
        return pcl.Particle(particle.getX(), particle.getY(), particle.getTheta(), particle.getWeight())

    def mcl(self, u, z, m):
        "augmented MCL function"
        particles = np.array([self.sample_motion_model_with_map(u, x_last, m) for x_last in self.particles])
        weights =  np.array([self.measurement_model(z, x_t, m) for x_t in particles])

        #keep this for now may not use
        for p, w in zip(particles, weights):
            p.setWeight(w)
        
        w_avg = np.mean(weights)
        if self.W_fast is None: #init as average of weights
            self.W_fast = self.W_slow = w_avg
        else:
            self.W_fast += self.fast_const*(w_avg - self.W_fast)
            self.W_slow += self.slow_const*(w_avg - self.W_slow)

        weights = np.cumsum(weights /np.sum(weights)) #smooth weights
        
        #redraw sample
        random_noise = np.random.rand(self.M) < max(0, 1- self.W_fast/self.W_slow)
        picks = np.random.rand(self.M)
        indices = np.searchsorted(weights, picks)

        #add random noise with propability p or redraw particle
        new_particles = [
            self.add_noise(m) if random_noise[i] else self.copy_particle(particles[indices[i]]) for i in range(self.M)]

        #set new belief distribution
        self.particles = np.array(new_particles)
        return
    
    def estimate_pose(self, particles = None):
        "return estimate of pose as average of particles, or given particle list"
        if particles is None:
            particles = self.particles
        return pcl.estimate_pose(particles)
    