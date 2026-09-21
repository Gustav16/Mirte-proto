#Mirte Proto code

import matplotlib.pyplot as plt


def plot_local_map(landmarks, mirte_radius=0.3, landmark_radius=0.5, ax=None, show=True):
    """
    Plot a local map.

    landmarks: list of [position, landmark_id] pairs, where position is an
               (x, z) array/tuple in meters, robot-centered — the format
               returned by LocalMap.get_map_from_mirte / LocalMap.landmarks.
    """
    if ax is None:
        _, ax = plt.subplots()

    # mirte sits at the origin, facing +z
    ax.add_patch(plt.Circle((0, 0), mirte_radius, color='blue', alpha=0.3))
    ax.plot(0, 0, 'b^', markersize=10, label='mirte')

    for position, landmark_id in landmarks:
        x, z = position
        ax.add_patch(plt.Circle((x, z), landmark_radius, color='red', alpha=0.2))
        ax.plot(x, z, 'ro')
        ax.annotate(str(landmark_id), (x, z), textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True)
    ax.legend(loc='upper right')

    if show:
        plt.show()

    return ax


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(
        os.path.join(os.path.dirname(__file__), '../../../Mirte/ku_mirte_python')
    )

    from ku_mirte import KU_Mirte
    from local_map import LocalMap

    mirte = KU_Mirte()
    local_map = LocalMap()
    local_map.update(mirte)

    plot_local_map(local_map.landmarks)

    mirte.close()
