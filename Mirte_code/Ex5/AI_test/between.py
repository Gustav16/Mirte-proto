import numpy as np


def landmark_dict(landmarks):
    points = {}

    for position, marker_id in landmarks:
        points[int(marker_id)] = np.asarray(
            position,
            dtype=float
        )

    return points


def midpoint(a, b):
    return (
        np.asarray(a, dtype=float)
        + np.asarray(b, dtype=float)
    ) / 2.0


def find_between_goal(
    local_map,
    target_ids=None,
    extra_margin=0.05
):
    """
    Find a goal position between two suitable ArUco landmarks.

    Returns:
        goal, (id_a, id_b)

    where goal is [x, z].
    """

    points = landmark_dict(local_map.landmarks)

    if len(points) < 2:
        print("Could not find at least two landmarks.")
        return None, None

    print("Detected ArUco landmarks:")

    for marker_id in sorted(points):
        x, z = points[marker_id]

        print(
            f"  ID {marker_id:3d}: "
            f"x = {x:+.3f} m, "
            f"z = {z:+.3f} m"
        )

    # --------------------------------------------------
    # Specific requested landmarks
    # --------------------------------------------------

    if target_ids is not None:
        id_a, id_b = target_ids

        if id_a not in points or id_b not in points:
            print(
                f"Could not find both requested IDs "
                f"{id_a} and {id_b}."
            )
            return None, None

        a = points[id_a]
        b = points[id_b]

        goal = midpoint(a, b)

        return goal, (id_a, id_b)

    # --------------------------------------------------
    # Exactly two landmarks
    # --------------------------------------------------

    ids = sorted(points)

    if len(ids) == 2:
        id_a, id_b = ids

        goal = midpoint(
            points[id_a],
            points[id_b]
        )

        return goal, (id_a, id_b)

    # --------------------------------------------------
    # More than two landmarks
    #
    # Sort landmarks left -> right and examine gaps
    # --------------------------------------------------

    ordered = sorted(
        points.items(),
        key=lambda item: item[1][0]
    )

    required_width = (
        2.0 * local_map.mirte_radius
        + 2.0 * local_map.landmark_radius
        + extra_margin
    )

    candidates = []

    for (id_a, a), (id_b, b) in zip(
        ordered[:-1],
        ordered[1:]
    ):
        gap_width = np.linalg.norm(b - a)

        if gap_width >= required_width:
            goal = midpoint(a, b)

            # Prefer a passage roughly in front of MIRTE.
            score = abs(goal[0])

            candidates.append(
                (score, goal, id_a, id_b)
            )

    if not candidates:
        print("Could not find a sufficiently wide passage.")
        return None, None

    candidates.sort(key=lambda candidate: candidate[0])

    _, goal, id_a, id_b = candidates[0]

    return goal, (id_a, id_b)

def goal_between_ids(local_map, id1, id2):
    """Return the midpoint between two specific ArUco landmarks."""
    landmarks = landmark_dict(local_map.landmarks)

    if id1 not in landmarks or id2 not in landmarks:
        return None

    return midpoint(landmarks[id1], landmarks[id2])


def goal_in_front_of_box(local_map, target_id, stop_distance=0.40):
    """
    Return a goal on the line from the robot start to a specific box,
    stopping stop_distance metres before the box.
    """
    landmarks = landmark_dict(local_map.landmarks)

    if target_id not in landmarks:
        return None

    box = np.asarray(landmarks[target_id], dtype=float)
    distance = np.linalg.norm(box)

    if distance <= stop_distance:
        return np.array([0.0, 0.0])

    return box * ((distance - stop_distance) / distance)



def find_safe_passage(local_map, extra_margin=0.05):
    """
    Find a sufficiently wide passage between two visible landmarks.

    Returns:
        goal, (id_a, id_b)

    The passage closest to straight ahead is preferred.
    """
    points = landmark_dict(local_map.landmarks)

    if len(points) < 2:
        return None, None

    required_width = (
        2.0 * local_map.mirte_radius
        + 2.0 * local_map.landmark_radius
        + extra_margin
    )

    candidates = []
    ids = list(points.keys())

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a = ids[i]
            id_b = ids[j]
            a = points[id_a]
            b = points[id_b]

            if np.linalg.norm(b - a) < required_width:
                continue

            goal = midpoint(a, b)

            # Only use passages in front of MIRTE.
            if goal[1] <= 0.10:
                continue

            # Prefer a passage close to the current forward direction.
            score = abs(goal[0])

            candidates.append(
                (score, goal, id_a, id_b)
            )

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item[0])
    _, goal, id_a, id_b = candidates[0]

    return goal, (id_a, id_b)


def limit_goal_distance(goal, max_distance):
    """Limit a local goal so MIRTE only explores a short distance at a time."""
    goal = np.asarray(goal, dtype=float)
    distance = np.linalg.norm(goal)

    if distance <= max_distance:
        return goal

    return goal * (max_distance / distance)
