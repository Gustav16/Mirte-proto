import numpy as np
import particle as pcl
import random_numbers as rn

class MCL:
    "MCL (monte-carlo localization) class"
    def __init__(self, 
                    prior,
                    # u is in cm and rad, so the alphas have units: variance = alpha * (cm or rad)^2
                    alpha1 = 0.01,  # rotation noise from rotation [rad^2/rad^2]: turn std = 10% of the turn TODO tune
                    alpha2 = 3e-7,  # rotation noise from translation [rad^2/cm^2]: ~3 deg heading std per metre TODO tune
                    alpha3 = 0.0025, # translation noise from translation [cm^2/cm^2]: drive std = 5% of the distance TODO tune
                    alpha4 = 0.5,  # translation noise from rotation [cm^2/rad^2]: ~1 cm std for a 90 deg turn TODO tune
                    fast_const = 0.3, #with 0.1/0.001 w_slow stayed at the poor first frames and no particles were ever injected
                    slow_const = 0.02, #0.3/0.02 reached the goal in 60/60 simulated runs (0.5/0.05 injected too often)
                    jitter_sigma = 2.0, # cm, std of position noise added after resampling
                    jitter_sigma_theta = 0.02, # rad, std of heading noise added after resampling
                    camera_offset = 0.0 # cm, camera in front of the robot centre (the particles are the robot centre)

                     ):
        self.alpha1 = alpha1  # rotation noise from rotation
        self.alpha2 = alpha2  # rotation noise from translation
        self.alpha3 = alpha3 # translation noise from translation  
        self.alpha4 = alpha4  # translation noise from rotation 
        self.W_fast = None
        self.W_slow = None
        self.fast_const = fast_const
        self.slow_const =  slow_const
        self.jitter_sigma = jitter_sigma
        self.jitter_sigma_theta = jitter_sigma_theta
        self.camera_offset = camera_offset
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
        particle_theta = x_t.getTheta()
        #the camera measures from its own position, camera_offset in front of the robot centre
        particle_x =  x_t.getX() + self.camera_offset*np.cos(particle_theta)
        particle_y =  x_t.getY() + self.camera_offset*np.sin(particle_theta)
        for pos, id in m.landmarks:
            landmark_x,landmark_y = pos
            delta = np.array([
            landmark_x - particle_x,
            landmark_y - particle_y])
            #heading 0 = +x, counter-clockwise positive, same as the motion model and selflocalize.py
            bearing = np.mod(np.arctan2(delta[1], delta[0]) - particle_theta + np.pi,
                             2 * np.pi) - np.pi
            res[id] = np.array([np.linalg.norm(delta), bearing]) #get distance and bearing to landmarks in a dictionary
        return res

    def p_hit(self, z_measured, z_true, sigma_hit, z_max):
        prop = 1

        for id in z_true:
            is_visible_from_pose = True
            measured = z_measured.get(id)

            if measured is None:
                if is_visible_from_pose:
                    prop *= 1
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
        sigma_hit=[10, 0.1745],   # std deviation of distance (cm) and bearing (rad, ~10 deg) TODO tune on robot
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
        "function for sampling motion model, u = [d_trans (cm), d_rot (rad)]: first turn d_rot on the spot, then drive d_trans straight"
        d_rot_1 = u[1]
        d_trans = u[0]

        #if no change add some random noise
        if d_rot_1 == 0.0 and d_trans == 0.0:
            x_t = [pcl.Particle(x_last.getX(), x_last.getY(), x_last.getTheta())]
            pcl.add_uncertainty(x_t, 5, 0.1745) #stddev of 10 cm, and stddev theta of about 10 degrees
            return x_t[0]
        d_rot_2 = 0 #we dont rotate after first rotation and transportation, but there may be noise

        d_rot_1_est = d_rot_1 + self.sample(self.alpha1*(d_rot_1**2)+ self.alpha2*(d_trans**2))
        d_trans_est = d_trans + self.sample(self.alpha3*(d_trans**2) + self.alpha4*(d_rot_1**2))
        d_rot_2_est = d_rot_2 + self.sample(self.alpha2*(d_trans**2))

        theta_new = x_last.getTheta() + d_rot_1_est #heading after the (noisy) first turn, d_rot_1_est already contains u[1]
        x_prime = x_last.getX() + d_trans_est*np.cos(theta_new)
        y_prime = x_last.getY() + d_trans_est*np.sin(theta_new)
        theta_prime = np.mod(theta_new + d_rot_2_est, 2.0 * np.pi)
        
        return pcl.Particle(x_prime, y_prime, theta_prime)

    def is_state_possible(self, x_t, m):
        return not m.in_collision((x_t.getX(), x_t.getY()))
    
    def sample_motion_model_with_map(self, u, x_last, m, max_tries = 10):
        "sampling with map"
        for _ in range(max_tries):
            x_t = self.sample_motion_model(u,x_last)
            if self.is_state_possible(x_t, m):
                return x_t
        #with u = 0 there is no motion noise, so a particle in collision (outside the map or in a landmark)
        #would be redrawn forever. Replace it with a random free pose instead
        return self.add_noise(m)
    
    def add_noise(self, m):
        "function for adding random pose to map"
        while True:
            x = np.random.uniform(m.map_area[0][0], m.map_area[1][0])
            y = np.random.uniform(m.map_area[0][1], m.map_area[1][1])
            theta = np.random.uniform(0, 2*np.pi)

            if not m.in_collision([x,y]):
                return pcl.Particle(x, y, theta, 1.0/self.M)

    def sample_from_measurement(self, z, m, sigma_dist = 5.0, sigma_angle = 0.05):
        """random pose that fits the measurement of one seen landmark (sensor resetting): at the measured distance
        in a random direction around it, turned so the landmark is at the measured bearing.
        One landmark only puts the robot on a circle around it, which uniform random particles almost never hit."""
        landmark_pos = {id: pos for pos, id in m.landmarks}
        seen = [id for id in z if id in landmark_pos]
        for _ in range(20):
            if not seen:
                break
            id = seen[np.random.randint(len(seen))]
            dist, bearing = z[id]
            landmark_x, landmark_y = landmark_pos[id]
            phi = np.random.uniform(0, 2*np.pi) #direction from the landmark to the camera
            d = dist + rn.randn(0.0, sigma_dist)
            theta = phi + np.pi - (bearing + rn.randn(0.0, sigma_angle)) #camera looks back along phi, rotated by the bearing
            x = landmark_x + d*np.cos(phi) - self.camera_offset*np.cos(theta) #robot centre is camera_offset behind the camera
            y = landmark_y + d*np.sin(phi) - self.camera_offset*np.sin(theta)
            if not m.in_collision([x,y]):
                return pcl.Particle(x, y, theta, 1.0/self.M)
        return self.add_noise(m)

    def copy_particle(self, particle):
        return pcl.Particle(particle.getX(), particle.getY(), particle.getTheta(), particle.getWeight())

    def mcl(self, u, z, m):
        "augmented MCL function"
        particles = np.array([self.sample_motion_model_with_map(u, x_last, m) for x_last in self.particles])
        weights =  np.array([self.measurement_model(z, x_t, m) for x_t in particles])

        #keep this for now may not use
        for p, w in zip(particles, weights):
            p.setWeight(w)
        
        #weights are a product over the seen landmarks, so their size jumps when the number of seen landmarks changes.
        #compare the average per seen landmark instead, so w_fast/w_slow only reacts to how well the particles fit
        n_seen = max(1, sum(1 for pos, id in m.landmarks if id in z))
        w_avg = np.mean(weights) ** (1.0 / n_seen)
        if self.W_fast is None: #init as average of weights
            self.W_fast = self.W_slow = w_avg
        else:
            self.W_fast += self.fast_const*(w_avg - self.W_fast)
            self.W_slow += self.slow_const*(w_avg - self.W_slow)

        weights = np.cumsum(weights /np.sum(weights)) #smooth weights

        print("w_avg:", w_avg, "W_fast:", self.W_fast, "W_slow:", self.W_slow,
              "p_random:", max(0, 1-self.W_fast/self.W_slow))
        
        #redraw sample
        random_noise = np.random.rand(self.M) < max(0, 1- self.W_fast/self.W_slow)
        picks = np.random.rand(self.M)
        indices = np.searchsorted(weights, picks)

        #add random noise with propability p or redraw particle
        new_particles = [
            self.sample_from_measurement(z, m) if random_noise[i] else self.copy_particle(particles[indices[i]]) for i in range(self.M)]

        #jitter so resampled copies don't collapse into one particle when standing still
        pcl.add_uncertainty(new_particles, self.jitter_sigma, self.jitter_sigma_theta)

        #set new belief distribution
        self.particles = np.array(new_particles)
        return
    
    def estimate_pose(self, particles = None):
        "return estimate of pose as average of particles, or given particle list"
        if particles is None:
            particles = self.particles
        return pcl.estimate_pose(particles)
    