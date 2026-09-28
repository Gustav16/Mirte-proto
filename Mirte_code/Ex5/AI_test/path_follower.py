import math
import numpy as np


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def closest_path_index(path, pose):
    distances = [
        np.linalg.norm(np.asarray(point) - pose[:2])
        for point in path
    ]
    return int(np.argmin(distances))


def lookahead_point(path, pose, lookahead=0.25):
    """
    Start at the closest path point and choose a point approximately
    lookahead metres farther along the path.
    """
    index = closest_path_index(path, pose)
    distance = 0.0

    for i in range(index, len(path) - 1):
        distance += np.linalg.norm(
            np.asarray(path[i + 1]) - np.asarray(path[i])
        )

        if distance >= lookahead:
            return np.asarray(path[i + 1], dtype=float)

    return np.asarray(path[-1], dtype=float)


def follow_path(
    path,
    mirte,
    localizer=None,
    get_observations=None,
    landmarks=None,
    linear_speed=0.30,
    angular_speed=0.70,
    linear_offset=-0.0355,
    lookahead=0.25,
    command_distance=0.08,
    goal_tolerance=0.08,
    max_steps=200,
):
    """
    Follow a start -> goal path.

    The controller repeatedly:
    1. Gets the current pose from MCL.
    2. Looks a little farther ahead on the path.
    3. Rotates toward that point.
    4. Drives a short distance.
    5. Updates MCL from the movement and new observations.
    """
    if path is None or len(path) < 2:
        return

    path = [np.asarray(point, dtype=float) for point in path]

    if localizer is None:
        pose = np.array([path[0][0], path[0][1], 0.0])
    else:
        pose = localizer.estimate_pose()

    for _ in range(max_steps):
        goal_distance = np.linalg.norm(path[-1] - pose[:2])

        if goal_distance <= goal_tolerance:
            print("Goal reached.")
            return

        target = lookahead_point(path, pose, lookahead)

        dx = target[0] - pose[0]
        dz = target[1] - pose[1]

        target_theta = math.atan2(-dx, dz)
        rotation = wrap_angle(target_theta - pose[2])

        # Rotate first, like the original Execute_path.
        if abs(rotation) > math.radians(5.0):
            scaler = 1.05 if rotation > 0 else 1.10

            mirte.drive(
                0.0,
                np.sign(rotation) * angular_speed,
                scaler * abs(rotation) / angular_speed,
            )

            if localizer is not None:
                observations = (
                    get_observations()
                    if get_observations is not None
                    else []
                )
                pose = localizer.update(
                    (0.0, rotation),
                    observations,
                    landmarks,
                )
            else:
                pose[2] = wrap_angle(pose[2] + rotation)

        # Recompute distance after rotation and move only a short step.
        distance_to_target = np.linalg.norm(target - pose[:2])
        distance = min(command_distance, distance_to_target, goal_distance)

        if distance <= 1e-6:
            continue

        mirte.drive(
            linear_speed,
            linear_offset,
            distance / linear_speed,
        )

        if localizer is not None:
            observations = (
                get_observations()
                if get_observations is not None
                else []
            )
            pose = localizer.update(
                (distance, 0.0),
                observations,
                landmarks,
            )
        else:
            pose[0] -= math.sin(pose[2]) * distance
            pose[1] += math.cos(pose[2]) * distance

        print(
            f"pose: x={pose[0]:+.2f}, "
            f"z={pose[1]:+.2f}, "
            f"theta={math.degrees(pose[2]):+.1f} deg"
        )

    print("Stopped because max_steps was reached.")
