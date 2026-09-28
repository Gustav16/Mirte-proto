import math
import numpy as np


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class MCL:
    """
    Small Monte Carlo Localization implementation.

    Particle format:
        [x, z, theta]

    Observation format:
        (marker_id, distance, bearing)
    """

    def __init__(
        self,
        local_map,
        number_of_particles=300,
        alpha_translation=0.05,
        alpha_rotation=0.05,
        sigma_range=0.10,
        sigma_bearing=math.radians(10.0),
    ):
        self.map = local_map
        self.number_of_particles = number_of_particles

        self.alpha_translation = alpha_translation
        self.alpha_rotation = alpha_rotation
        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing

        self.particles = self._make_initial_particles()
        self.weights = np.ones(number_of_particles) / number_of_particles

    def _make_initial_particles(self):
        """
        The local map starts with MIRTE at [0, 0].
        Use a small uncertainty around this initial pose.
        """
        particles = []

        while len(particles) < self.number_of_particles:
            x = np.random.normal(0.0, 0.05)
            z = np.random.normal(0.0, 0.05)
            theta = np.random.normal(0.0, math.radians(5.0))

            if not self.map.in_collision(np.array([x, z])):
                particles.append([x, z, theta])

        return np.asarray(particles, dtype=float)

    def sample_motion_model(self, movement):
        """
        Move every particle according to the commanded movement.

        movement = (translation, rotation)

        rotation is applied first, then translation.
        theta = 0 means forward in +z.
        """
        translation, rotation = movement
        new_particles = []

        for x, z, theta in self.particles:
            noisy_rotation = rotation + np.random.normal(
                0.0,
                self.alpha_rotation * abs(rotation) + 0.005,
            )
            noisy_translation = translation + np.random.normal(
                0.0,
                self.alpha_translation * abs(translation) + 0.002,
            )

            new_theta = wrap_angle(theta + noisy_rotation)
            new_x = x - math.sin(new_theta) * noisy_translation
            new_z = z + math.cos(new_theta) * noisy_translation

            if self.map.in_collision(np.array([new_x, new_z])):
                new_x, new_z = x, z

            new_particles.append([new_x, new_z, new_theta])

        self.particles = np.asarray(new_particles, dtype=float)

    def measurement_model(self, observations, landmarks):
        """
        Give high weight to particles whose expected ArUco distance
        and bearing are close to the camera observations.
        """
        if not observations:
            self.weights = np.ones(self.number_of_particles)
            return

        landmark_dict = {
            int(marker_id): np.asarray(position, dtype=float)
            for position, marker_id in landmarks
        }

        weights = []

        for x, z, theta in self.particles:
            weight = 1.0

            for marker_id, measured_range, measured_bearing in observations:
                marker_id = int(marker_id)

                if marker_id not in landmark_dict:
                    continue

                lx, lz = landmark_dict[marker_id]
                dx = lx - x
                dz = lz - z

                expected_range = math.hypot(dx, dz)
                expected_bearing = wrap_angle(math.atan2(-dx, dz) - theta)

                range_error = measured_range - expected_range
                bearing_error = wrap_angle(measured_bearing - expected_bearing)

                range_probability = math.exp(
                    -0.5 * (range_error / self.sigma_range) ** 2
                )
                bearing_probability = math.exp(
                    -0.5 * (bearing_error / self.sigma_bearing) ** 2
                )

                weight *= max(
                    range_probability * bearing_probability,
                    1e-12,
                )

            weights.append(weight)

        self.weights = np.asarray(weights, dtype=float)

    def resample(self):
        total = np.sum(self.weights)

        if total <= 0 or not np.isfinite(total):
            probabilities = np.ones(self.number_of_particles)
            probabilities /= self.number_of_particles
        else:
            probabilities = self.weights / total

        indices = np.random.choice(
            self.number_of_particles,
            size=self.number_of_particles,
            replace=True,
            p=probabilities,
        )

        self.particles = self.particles[indices]
        self.weights = np.ones(self.number_of_particles)
        self.weights /= self.number_of_particles

    def estimate_pose(self):
        x = np.mean(self.particles[:, 0])
        z = np.mean(self.particles[:, 1])

        sin_mean = np.mean(np.sin(self.particles[:, 2]))
        cos_mean = np.mean(np.cos(self.particles[:, 2]))
        theta = math.atan2(sin_mean, cos_mean)

        return np.array([x, z, theta])

    def update(self, movement, observations, landmarks):
        """
        One complete MCL update:
        motion -> measurement -> resampling -> pose estimate
        """
        self.sample_motion_model(movement)
        self.measurement_model(observations, landmarks)
        self.resample()
        return self.estimate_pose()
