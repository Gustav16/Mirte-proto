import numpy as np


def segment_is_free(a, b, local_map, step=0.03):
    """Check points along a straight line for collisions."""
    if hasattr(local_map,"segment_is_free"):
        return local_map.segment_is_free(a,b)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    distance = np.linalg.norm(b - a)
    if distance <= 1e-9:
        return not local_map.in_collision(a)

    n = max(1, int(np.ceil(distance / step)))

    for i in range(n + 1):
        point = a + (b - a) * (i / n)
        if local_map.in_collision(point):
            return False

    return True


def shortcut_path(path, local_map):
    """
    Remove unnecessary waypoints.
    Input and output are start -> goal.
    """
    if path is None:
        return None
    if len(path) <= 2:
        if len(path)==2 and not segment_is_free(path[0],path[1],local_map):
            raise ValueError("Input path contains a colliding segment.")
        return path

    result = [np.asarray(path[0], dtype=float)]
    current = 0

    while current < len(path) - 1:
        # Try the farthest point first.
        next_index = len(path) - 1

        while next_index > current + 1:
            if segment_is_free(path[current], path[next_index], local_map):
                break
            next_index -= 1

        if not segment_is_free(path[current],path[next_index],local_map):
            raise ValueError("Input path contains a colliding segment.")
        result.append(np.asarray(path[next_index], dtype=float))
        current = next_index

    return result


def add_path_points(path, spacing=0.05):
    """Add evenly spaced reference points along each straight segment."""
    if path is None or len(path) < 2:
        return path

    result = [np.asarray(path[0], dtype=float)]

    for a, b in zip(path[:-1], path[1:]):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        distance = np.linalg.norm(b - a)
        n = max(1, int(np.ceil(distance / spacing)))

        for i in range(1, n + 1):
            result.append(a + (b - a) * (i / n))

    return result


def smooth_path(path, local_map, spacing=0.05):
    """
    Simple smoothing used by the main program:
    1. Remove unnecessary RRT waypoints.
    2. Add evenly spaced reference points.
    """
    if path is None:
        return None

    short_path = shortcut_path(path, local_map)
    return add_path_points(short_path, spacing)
