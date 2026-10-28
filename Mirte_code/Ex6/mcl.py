import numpy as np
import particle as pcl
import random_numbers as rn

#delte
from time import perf_counter

class MCL:
    "MCL (monte-carlo localization) class"
    def __init__(self, 
                    prior,
                    # u is in cm and rad, so the alphas have units: variance = alpha * (cm or rad)^2
                    alpha1 = 0.01,  # rotation noise from rotation [rad^2/rad^2]: turn std = 10% of the turn TODO tune
                    alpha2 = 3e-7,  # rotation noise from translation [rad^2/cm^2]: ~3 deg heading std per metre TODO tune
                    alpha3 = 0.0025, # translation noise from translation [cm^2/cm^2]: drive std = 5% of the distance TODO tune
                    alpha4 = 0.5,  # translation noise from rotation [cm^2/rad^2]: ~1 cm std for a 90 deg turn TODO tune
                    fast_const = 0.3, #with 0.1/0.001 w_slow stayed at the poor first frames and no particles were ever injected #TODO increse fast so less random noise is injected
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
        for pos, radius, id in m.landmarks:
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

    @staticmethod
    def _in_collision_fast(pos, m):
        """Equivalent to LocalMap.in_collision for the Ex6 circular map.

        Uses scalar squared distances instead of allocating NumPy arrays for
        each particle.  Map boundaries, landmark radii, obstacle radii and
        the <= collision threshold are unchanged.  The original map remains
        untouched.  See test_mcl_step3.py for equivalence checks.
        """
        # Preserve compatibility with other map implementations that expose
        # in_collision() but not the Ex6 map attributes.
        if not (hasattr(m, 'map_area') and hasattr(m, 'landmarks')
                and hasattr(m, 'obstacles') and hasattr(m, 'mirte_radius')
                and hasattr(m, 'obstacale_radius')):
            return bool(m.in_collision(pos))

        x = float(pos[0])
        y = float(pos[1])
        low, high = m.map_area

        # Exactly the same inclusive/exclusive bounds as LocalMap.in_collision:
        # outside is forbidden; points exactly on the boundary are allowed.
        if x < low[0] or y < low[1] or x > high[0] or y > high[1]:
            return True

        robot_radius = m.mirte_radius
        for landmark, radius, _landmark_id in m.landmarks:
            dx = x - landmark[0]
            dy = y - landmark[1]
            if dx * dx + dy * dy <= (robot_radius + radius) ** 2:
                return True

        clearance_sq = (m.obstacale_radius + robot_radius) ** 2
        for obstacle, _obstacle_id in m.obstacles:
            dx = x - obstacle[0]
            dy = y - obstacle[1]
            if dx * dx + dy * dy <= clearance_sq:
                return True

        return False

    # STEP 4: BATCH CALCULATIONS (same mathematical model, NumPy arrays)
    @staticmethod
    def _collision_mask_batch(positions, m):
        """Collision for an (N,2) array, matching _in_collision_fast for Ex6.

        For other map types, use the map's own collision method for safety.
        """
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must be an (N, 2) array")

        if not (hasattr(m, 'map_area') and hasattr(m, 'landmarks')
                and hasattr(m, 'obstacles') and hasattr(m, 'mirte_radius')
                and hasattr(m, 'obstacale_radius')):
            return np.fromiter((bool(m.in_collision(xy)) for xy in positions),
                               dtype=bool, count=len(positions))

        x = positions[:, 0]
        y = positions[:, 1]
        low, high = m.map_area
        collided = ((x < low[0]) | (y < low[1]) |
                    (x > high[0]) | (y > high[1]))

        robot_radius = m.mirte_radius
        for landmark, radius, _landmark_id in m.landmarks:
            dx = x - landmark[0]
            dy = y - landmark[1]
            collided |= (dx * dx + dy * dy <= (robot_radius + radius) ** 2)

        clearance_sq = (m.obstacale_radius + robot_radius) ** 2
        for obstacle, _obstacle_id in m.obstacles:
            dx = x - obstacle[0]
            dy = y - obstacle[1]
            collided |= (dx * dx + dy * dy <= clearance_sq)

        return collided

    def _sample_motion_batch(self, u, m):
        """Sample motion for every particle, retry collisions up to 10 times.

        Each particle uses its ORIGINAL pose on every retry, as in the
        original sample_motion_model_with_map. Failed particles are reset
        to a random collision-free pose, just like add_noise().

        Independent Gaussian samples have the same distributions as the
        scalar implementation, but a different order of random draws.
        """
        poses = np.array([(p.getX(), p.getY(), p.getTheta())
                          for p in self.particles], dtype=float)
        result = np.empty_like(poses)
        pending = np.arange(len(poses))
        d_trans = float(u[0])
        d_rot = float(u[1])
        stationary = (d_trans == 0.0 and d_rot == 0.0)

        if not stationary:
            sigma_rot1 = np.sqrt(self.alpha1 * d_rot**2 + self.alpha2 * d_trans**2)
            sigma_trans = np.sqrt(self.alpha3 * d_trans**2 + self.alpha4 * d_rot**2)
            sigma_rot2 = np.sqrt(self.alpha2 * d_trans**2)

        for _attempt in range(10):
            if not len(pending):
                break
            old = poses[pending]
            count = len(pending)
            if stationary:
                # pcl.add_uncertainty: independent x/y/theta Gaussian noise.
                x = old[:, 0] + 3.0 * np.random.randn(count)
                y = old[:, 1] + 3.0 * np.random.randn(count)
                theta = np.mod(old[:, 2] + 0.1045 * np.random.randn(count),
                               2.0 * np.pi)
            else:
                # Exact rotate-then-translate motion-model equations.
                rot1 = d_rot + np.random.normal(0.0, sigma_rot1, count)
                trans = d_trans + np.random.normal(0.0, sigma_trans, count)
                rot2 = np.random.normal(0.0, sigma_rot2, count)
                theta_mid = old[:, 2] + rot1
                x = old[:, 0] + trans * np.cos(theta_mid)
                y = old[:, 1] + trans * np.sin(theta_mid)
                theta = np.mod(theta_mid + rot2, 2.0 * np.pi)

            candidate = np.column_stack((x, y, theta))
            invalid = self._collision_mask_batch(candidate[:, :2], m)
            result[pending[~invalid]] = candidate[~invalid]
            pending = pending[invalid]

        # The original add_noise() samples uniform poses until they are free.
        # Batch only the particles which exceeded the 10 attempts.
        while len(pending):
            count = len(pending)
            candidate = np.column_stack((
                np.random.uniform(m.map_area[0][0], m.map_area[1][0], count),
                np.random.uniform(m.map_area[0][1], m.map_area[1][1], count),
                np.random.uniform(0.0, 2.0 * np.pi, count),
            ))
            invalid = self._collision_mask_batch(candidate[:, :2], m)
            result[pending[~invalid]] = candidate[~invalid]
            pending = pending[invalid]

        return result

    def _measurement_batch(self, z, poses, m):
        """Same p_hit likelihood as the original, for all particles at once.

        The original p_hit multiplies by 1 for non-observed landmarks. Hence
        only observed landmark IDs are computed here. The Gaussian noise and
        angular wrapping are unchanged.
        """
        x = poses[:, 0]
        y = poses[:, 1]
        theta = poses[:, 2]
        if self.camera_offset:
            cam_x = x + self.camera_offset * np.cos(theta)
            cam_y = y + self.camera_offset * np.sin(theta)
        else:
            cam_x, cam_y = x, y
        weights = np.ones(len(poses), dtype=float)

        # Match the landmark iteration order of true_z_for_landmarks/p_hit.
        for pos, _radius, landmark_id in m.landmarks:
            measured = z.get(landmark_id)
            if measured is None:
                continue
            dx = pos[0] - cam_x
            dy = pos[1] - cam_y
            expected_dist = np.sqrt(dx * dx + dy * dy)
            expected_bearing = np.mod(np.arctan2(dy, dx) - theta + np.pi,
                                     2.0 * np.pi) - np.pi
            bearing_error = np.mod(measured[1] - expected_bearing + np.pi,
                                   2.0 * np.pi) - np.pi
            weights *= self.gaussian_pdf(measured[0], expected_dist, 10)
            weights *= self.gaussian_pdf(bearing_error, 0, 0.1745)
        return weights

    def is_state_possible(self, x_t, m):
        return not self._in_collision_fast((x_t.getX(), x_t.getY()), m)
    
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

            if not self._in_collision_fast((x, y), m):
                return pcl.Particle(x, y, theta, 1.0/self.M)

    def sample_from_measurement(self, z, m, sigma_dist = 5.0, sigma_angle = 0.05):
        """random pose that fits the measurement of one seen landmark (sensor resetting): at the measured distance
        in a random direction around it, turned so the landmark is at the measured bearing.
        One landmark only puts the robot on a circle around it, which uniform random particles almost never hit."""
        landmark_pos = {id: pos for pos, radius, id in m.landmarks}
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
            
            return pcl.Particle(x, y, theta, 1.0/self.M)

    def copy_particle(self, particle):
        return pcl.Particle(particle.getX(), particle.getY(), particle.getTheta(), particle.getWeight())

    def mcl(self, u, z, m):
        """Augmented MCL using vectorized motion and measurement equations."""
        t0 = perf_counter()
        poses = self._sample_motion_batch(u, m)
        t1 = perf_counter()
        weights = self._measurement_batch(z, poses, m)
        t2 = perf_counter()

        print(f"Motion: {t1-t0:.3f}s | Measurements: {t2-t1:.3f}s")

        # Keep the same adaptive W_fast/W_slow update and resampling rule.
        n_seen = max(1, sum(1 for pos, radius, landmark_id in m.landmarks
                            if landmark_id in z))
        w_avg = np.mean(weights) ** (1.0 / n_seen)
        if self.W_fast is None:
            self.W_fast = self.W_slow = w_avg
        else:
            self.W_fast += self.fast_const * (w_avg - self.W_fast)
            self.W_slow += self.slow_const * (w_avg - self.W_slow)

        cumulative = np.cumsum(weights / np.sum(weights))
        p_random = max(0, 1 - self.W_fast / self.W_slow)
        print("w_avg:", w_avg, "W_fast:", self.W_fast, "W_slow:", self.W_slow,
              "p_random:", p_random)

        random_noise = np.random.rand(self.M) < p_random
        picks = np.random.rand(self.M)
        indices = np.searchsorted(cumulative, picks)

        # The scalar implementation copies the selected Particle including
        # its UNNORMALIZED measurement weight. Preserve that behavior here.
        resampled_poses = poses[indices].copy()
        resampled_weights = weights[indices].copy()
        for i in np.flatnonzero(random_noise):
            replacement = self.sample_from_measurement(z, m)
            # Preserve the original failure behavior for absent landmark IDs.
            if replacement is None:
                raise ValueError("sensor-reset had no known landmark to sample from")
            resampled_poses[i] = (replacement.getX(), replacement.getY(),
                                  replacement.getTheta())
            resampled_weights[i] = replacement.getWeight()

        self.particles = np.array([
            pcl.Particle(x, y, theta, weight)
            for (x, y, theta), weight in zip(resampled_poses, resampled_weights)
        ])
        return

    def estimate_pose(self, particles = None):
        "return estimate of pose as average of particles, or given particle list"
        if particles is None:
            particles = self.particles
        return pcl.estimate_pose(particles)


    #added may change
    def pose_variance(est_pose, particles_list):
        """Calculate the variance of x, y, and theta across particles."""
        if len(particles_list) == 0:
            return 0.0, 0.0, 0.0

        # Estimate the mean pose
        mean_pose = est_pose

        x_mean = mean_pose.getX()
        y_mean = mean_pose.getY()
        theta_mean = mean_pose.getTheta()

        # Extract particle poses
        x = np.array([p.getX() for p in particles_list])
        y = np.array([p.getY() for p in particles_list])
        theta = np.array([p.getTheta() for p in particles_list])

        # Position variances
        x_var = np.mean((x - x_mean) ** 2)
        y_var = np.mean((y - y_mean) ** 2)

        # Angular variance using wrapped angular differences
        theta_diff = np.arctan2(
            np.sin(theta - theta_mean),
            np.cos(theta - theta_mean)
        )
        theta_var = np.mean(theta_diff ** 2)

        return x_var, y_var, theta_var

    def has_converged(est_pose, mcl, x_var_err, y_var_err, theta_var_err) -> bool:
        "method for checking convergence"
        x_var, y_var, theta_var = MCL.pose_variance(est_pose, mcl.particles)

        return (x_var < x_var_err
            and y_var < y_var_err
            and theta_var < theta_var_err
        )
    
    ### Updating MCL ###
    def update_mcl(mcl, u, z, global_map, NUM_PARTICLES) -> None:
        "function for updating MCL"
        if z:
            mcl.mcl(u,z,global_map)
        else: #TODO look at this
            if u[0] != 0.0 or u[1] != 0.0: # without a move or a measurement nothing changed (and the motion model adds noise for u = 0)
                poses = mcl._sample_motion_batch(u, global_map)
                mcl.particles = np.array([
                    pcl.Particle(x, y, theta, 1.0 / NUM_PARTICLES)
                    for x, y, theta in poses
                ])
            else:
                for p in mcl.particles:
                    p.setWeight(1.0 / NUM_PARTICLES)