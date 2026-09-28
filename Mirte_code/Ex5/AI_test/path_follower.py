import math
import time
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
    Choose a point farther ahead on the path.
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
    linear_speed=0.28,
    max_angular_speed=0.75,
    lookahead=0.25,
    goal_tolerance=0.08,
    control_period=0.25,
    heading_gain=1.8,
    max_time=30.0,
):
    """
    Smooth path follower.

    Important difference from the old version:
    MIRTE is NOT told to drive 8 cm and stop.

    Instead it receives a continuous non-blocking velocity command.
    While it is still moving, the controller updates the steering
    direction and MCL estimate.

    The robot only stops when the goal is reached or the safety
    timeout is reached.
    """
    if path is None or len(path) < 2:
        return

    path = [
        np.asarray(point, dtype=float)
        for point in path
    ]

    if localizer is None:
        pose = np.array([
            path[0][0],
            path[0][1],
            0.0
        ])
    else:
        pose = localizer.estimate_pose()

    start_time = time.time()
    last_time = start_time

    print("Starting smooth path following...")

    try:
        while time.time() - start_time < max_time:
            goal_distance = np.linalg.norm(
                path[-1] - pose[:2]
            )

            if goal_distance <= goal_tolerance:
                mirte.stop()
                print("Goal reached.")
                return

            target = lookahead_point(
                path,
                pose,
                lookahead
            )

            dx = target[0] - pose[0]
            dz = target[1] - pose[1]

            target_theta = math.atan2(-dx, dz)
            heading_error = wrap_angle(
                target_theta - pose[2]
            )

            # Proportional steering.
            angular = heading_gain * heading_error
            angular = float(np.clip(
                angular,
                -max_angular_speed,
                max_angular_speed
            ))

            # Slow down on sharper turns, but keep moving.
            turn_amount = min(
                abs(heading_error) / math.radians(45.0),
                1.0
            )

            speed = linear_speed * (
                1.0 - 0.55 * turn_amount
            )

            # Very close to the goal: approach more slowly.
            if goal_distance < 0.20:
                speed = min(speed, 0.18)

            # Continuous command.
            # duration=None means keep driving until a new command arrives.
            # blocking=False means Python can immediately continue.
            mirte.drive(
                speed,
                angular,
                None,
                blocking=False,
                interrupt=True,
            )

            now = time.time()
            dt = max(now - last_time, control_period)
            last_time = now

            translation = speed * dt
            rotation = angular * dt

            if localizer is not None:
                observations = (
                    get_observations()
                    if get_observations is not None
                    else []
                )

                pose = localizer.update(
                    (translation, rotation),
                    observations,
                    landmarks,
                )
            else:
                pose[2] = wrap_angle(
                    pose[2] + rotation
                )
                pose[0] -= (
                    math.sin(pose[2]) * translation
                )
                pose[1] += (
                    math.cos(pose[2]) * translation
                )

            print(
                f"pose: x={pose[0]:+.2f}, "
                f"z={pose[1]:+.2f}, "
                f"theta={math.degrees(pose[2]):+.1f} deg"
            )

            # KU_Mirte.drive already spends about 0.2 s submitting
            # the command. Only sleep if the loop was faster.
            elapsed = time.time() - now
            if elapsed < control_period:
                time.sleep(control_period - elapsed)

        mirte.stop()
        print("Stopped because path-following timeout was reached.")

    except BaseException:
        mirte.stop()
        raise
