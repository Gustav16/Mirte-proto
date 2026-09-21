import numpy as np





class ParticleFilter:

    def __init__(self, N, initial_pose,
                    distance_std=0.01, # get estimate for std
                    angle_std=np.deg2rad(2) #get estimate for std
                    ):

        self.N = N
        self.distance_std = distance_std
        self.angle_std = angle_std

        self.belief = initialize_particles(N, initial_pose)
        self.weights = np.ones(N) / N

    def update(self, distance, dtheta, observations, landmarks):

        # belief at time t-1
        previous_belief = self.belief

        # 1. Predict: p(x_t | x_{t-1}, u_t)
        predicted_belief = self.motion_update(
            previous_belief, distance, dtheta
        )

        # 2. Correct: p(z_t | x_t)
        weights = self.sensor_update(
            predicted_belief, observations, landmarks
        )

        # 3. Resample predicted belief according to weights
        self.belief = self.resample(
            predicted_belief, weights
        )

    def motion_update(self, belief, distance, dtheta):

        predicted_belief = belief.copy()

        noisy_distance = (
            distance +
            np.random.normal(0, self.distance_std, self.N)
        )

        noisy_dtheta = (
            dtheta +
            np.random.normal(0, self.angle_std, self.N)
        )

        predicted_belief[:, 2] += noisy_dtheta

        predicted_belief[:, 0] += (
            noisy_distance *
            np.cos(predicted_belief[:, 2])
        )

        predicted_belief[:, 1] += (
            noisy_distance *
            np.sin(predicted_belief[:, 2])
        )

        return predicted_belief

    def sensor_update(self, belief, observations, landmarks):

        weights = np.zeros(self.N)

        for i, particle in enumerate(belief):

            # What would Mirte see from this particle?
            predicted = ...

            # Compare predicted observation to actual observation
            error = ...

            # Likelihood of observing this error
            weights[i] = ...

    weights /= np.sum(weights)

    return weights

    def resample(self, belief, weights):

        weights = weights / np.sum(weights)
        cumulative_weights = np.cumsum(weights)

        picks = np.random.rand(self.N)

        indices = np.searchsorted(
            cumulative_weights,
            picks
        )
        return belief[indices].copy()

    def get_pose(self):
        x = np.mean(self.belief[:, 0])
        y = np.mean(self.belief[:, 1])

        # Circular mean for theta
        theta = np.arctan2(
            np.mean(np.sin(self.belief[:, 2])),
            np.mean(np.cos(self.belief[:, 2]))
        )

        return np.array([x, y, theta])