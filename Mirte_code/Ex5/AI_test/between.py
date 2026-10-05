# between.py
#
# Goal selection using the oriented rectangular box geometry from local_map.py.
#
# IMPORTANT:
# This file does NOT interpret ArUco rvec/tvec itself.
# local_map.py is the single source of truth for box pose and geometry.

import numpy as np
from geometry_utils import rectangle_distance


def midpoint(a, b):
    return (
        np.asarray(a, dtype=float)
        + np.asarray(b, dtype=float)
    ) / 2.0


def _box(local_map, marker_id):
    """Get one box through LocalMap's public geometry interface."""
    return local_map.get_box(int(marker_id))


def _closest_points_between_boxes(box_a, box_b):
    """
    Approximate the relevant facing points between two oriented rectangles.

    We use support points in the direction from one box center to the other.
    For a rectangle, the support point is the extreme point in a direction.
    """
    center_a = box_a["box_center"]
    center_b = box_b["box_center"]

    delta = center_b - center_a
    distance = np.linalg.norm(delta)

    if distance < 1e-9:
        return None, None

    direction = delta / distance

    def support(box, direction_vector):
        half_w = box["width"] / 2.0
        half_d = box["depth"] / 2.0

        sw = np.sign(np.dot(direction_vector, box["width_direction"]))
        sd = np.sign(np.dot(direction_vector, box["depth_direction"]))

        return (
            box["box_center"]
            + sw * half_w * box["width_direction"]
            + sd * half_d * box["depth_direction"]
        )

    point_a = support(box_a, direction)
    point_b = support(box_b, -direction)

    return point_a, point_b


def passage_between_ids(
    local_map,
    id_a,
    id_b,
    extra_margin=0.05
):
    """
    Compute a passage candidate between two oriented boxes.

    Returns:
        goal, free_width

    goal is the midpoint between the two facing boundary points.
    free_width is the exact shortest rectangle-to-rectangle distance.

    The passage is accepted only if MIRTE's diameter plus extra_margin fits.
    """
    box_a = _box(local_map, id_a)
    box_b = _box(local_map, id_b)

    if box_a is None or box_b is None:
        return None, None

    point_a, point_b = _closest_points_between_boxes(
        box_a,
        box_b
    )

    if point_a is None:
        return None, None

    free_width = rectangle_distance(box_a, box_b)

    required_width = (
        2.0 * local_map.mirte_radius
        + float(extra_margin)
    )

    if free_width < required_width:
        return None, free_width

    goal = midpoint(point_a, point_b)

    # A candidate goal must itself be collision-free with every visible box,
    # including any third box.
    if local_map.in_collision(goal):
        return None, free_width
    required_clearance = local_map.mirte_radius + max(extra_margin / 2, local_map.clearance_margin)
    if any(local_map.distance_to_box(goal,i) <= required_clearance for i in local_map.get_visible_box_ids()):
        return None, free_width

    return goal, free_width


def goal_between_ids(
    local_map,
    id1,
    id2,
    extra_margin=0.05
):
    """Return a safe goal in the physical passage between two boxes."""
    goal, _ = passage_between_ids(
        local_map,
        id1,
        id2,
        extra_margin=extra_margin
    )
    return goal


def find_between_goal(
    local_map,
    target_ids=None,
    extra_margin=0.05
):
    """
    Find a safe passage goal.

    If target_ids is supplied, only that pair is considered.
    Otherwise all visible box pairs are evaluated and the passage closest
    to MIRTE's forward direction is preferred.
    """
    ids = local_map.get_visible_box_ids()

    if len(ids) < 2:
        print("Could not find at least two boxes.")
        return None, None

    print("Detected boxes:")
    for marker_id in ids:
        box = local_map.get_box(marker_id)
        center = box["box_center"]
        print(
            f"  ID {marker_id:3d}: "
            f"center=({center[0]:+.3f}, {center[1]:+.3f}) m, "
            f"orientation={box['theta_deg']:+.1f} deg"
        )

    if target_ids is not None:
        id_a, id_b = map(int, target_ids)

        goal, free_width = passage_between_ids(
            local_map,
            id_a,
            id_b,
            extra_margin=extra_margin
        )

        if goal is None:
            if free_width is None:
                print(f"Could not find both requested boxes {id_a} and {id_b}.")
            else:
                print(
                    f"Passage between {id_a} and {id_b} is too narrow "
                    f"or blocked. Estimated free width: {free_width:.3f} m."
                )
            return None, None

        return goal, (id_a, id_b)

    return find_safe_passage(
        local_map,
        extra_margin=extra_margin
    )


def find_safe_passage(local_map, extra_margin=0.05):
    """
    Evaluate every pair of visible oriented boxes.

    Candidate requirements:
      - enough physical free width for MIRTE + margin
      - goal is in front of MIRTE
      - goal is collision-free with ALL visible boxes

    The candidate closest to straight ahead is preferred.
    """
    ids = local_map.get_visible_box_ids()

    if len(ids) < 2:
        return None, None

    candidates = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a = ids[i]
            id_b = ids[j]

            goal, free_width = passage_between_ids(
                local_map,
                id_a,
                id_b,
                extra_margin=extra_margin
            )

            if goal is None:
                continue

            if goal[1] <= 0.10:
                continue

            # Prefer a passage close to MIRTE's current forward axis.
            # Secondary preference: wider passage.
            score = (
                abs(goal[0]),
                -free_width
            )

            candidates.append(
                (score, goal, id_a, id_b, free_width)
            )

    if not candidates:
        print("Could not find a sufficiently wide, collision-free passage.")
        return None, None

    candidates.sort(key=lambda item: item[0])
    _, goal, id_a, id_b, free_width = candidates[0]

    print(
        f"Selected passage between ID {id_a} and ID {id_b}: "
        f"goal=({goal[0]:+.3f}, {goal[1]:+.3f}) m, "
        f"free width={free_width:.3f} m"
    )

    return goal, (id_a, id_b)


def _ray_box_surface_distance(box, direction):
    """
    Distance from box center to its boundary along a given unit direction.

    The direction is expressed in the global x-z map.
    """
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)

    if norm < 1e-9:
        return None

    direction = direction / norm

    du = abs(np.dot(direction, box["width_direction"]))
    dv = abs(np.dot(direction, box["depth_direction"]))

    candidates = []

    if du > 1e-9:
        candidates.append(
            (box["width"] / 2.0) / du
        )

    if dv > 1e-9:
        candidates.append(
            (box["depth"] / 2.0) / dv
        )

    if not candidates:
        return None

    return min(candidates)


def goal_in_front_of_box(
    local_map,
    target_id,
    stop_distance=0.40
):
    """
    Goal on the robot->box line, stop_distance BEFORE the physical box surface.

    This differs from the old implementation, which stopped relative to the
    box center and therefore ignored box size/orientation.
    """
    box = _box(local_map, target_id)

    if box is None:
        return None

    center = np.asarray(box["box_center"], dtype=float)
    center_distance = np.linalg.norm(center)

    if center_distance < 1e-9:
        return np.array([0.0, 0.0])

    robot_to_box = center / center_distance

    # From the box center, the robot lies in the opposite direction.
    surface_radius = _ray_box_surface_distance(
        box,
        -robot_to_box
    )

    if surface_radius is None:
        return None

    surface_distance_from_robot = (
        center_distance - surface_radius
    )

    desired_distance = (
        surface_distance_from_robot
        - float(stop_distance)
    )

    if desired_distance <= 0.0:
        return np.array([0.0, 0.0])

    goal = robot_to_box * desired_distance

    # Goal should not itself collide with another box.
    if local_map.in_collision(goal):
        return None

    return goal


def limit_goal_distance(goal, max_distance):
    """Limit a local goal so MIRTE explores only max_distance at a time."""
    if goal is None:
        return None

    goal = np.asarray(goal, dtype=float)
    distance = np.linalg.norm(goal)

    if distance <= max_distance:
        return goal

    if distance < 1e-9:
        return goal

    return goal * (float(max_distance) / distance)


def landmark_dict(landmarks):
    """Legacy list to {id: position}; new geometry code uses LocalMap.boxes."""
    return {int(i): np.asarray(p,float).copy() for p,i in landmarks}
