import unittest
from pathlib import Path
from types import SimpleNamespace


class TargetReachedTests(unittest.TestCase):
    def setUp(self):
        self.target_id = 6
        self.landmarks = {self.target_id: (300.0, 400.0)}
        self.threshold_cm = 60.0

    def is_target_reached(self, observations, pose, has_converged):
        if (
            self.target_id in observations
            and observations[self.target_id][0] <= self.threshold_cm
        ):
            return True

        if has_converged:
            robot_pos = (pose.getX(), pose.getY())
            target_pos = self.landmarks[self.target_id]
            distance_squared = sum(
                (robot - target) ** 2
                for robot, target in zip(robot_pos, target_pos)
            )
            return distance_squared <= self.threshold_cm**2

        return False

    def test_camera_observation_within_threshold_reaches_target(self):
        self.assertTrue(
            self.is_target_reached({self.target_id: [59.9, 0.0]}, None, False)
        )

    def test_camera_observation_at_threshold_reaches_target(self):
        self.assertTrue(
            self.is_target_reached(
                {self.target_id: [self.threshold_cm, 0.0]}, None, False
            )
        )

    def test_camera_observation_outside_threshold_does_not_reach_target(self):
        self.assertFalse(
            self.is_target_reached({self.target_id: [60.1, 0.0]}, None, False)
        )

    def test_converged_pose_within_threshold_reaches_target(self):
        pose = SimpleNamespace(getX=lambda: 300.0, getY=lambda: 450.0)
        self.assertTrue(
            self.is_target_reached({}, pose, True)
        )

    def test_unconverged_pose_does_not_reach_target(self):
        pose = SimpleNamespace(getX=lambda: 300.0, getY=lambda: 450.0)
        self.assertFalse(
            self.is_target_reached({}, pose, False)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
