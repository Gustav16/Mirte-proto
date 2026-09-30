# local_map.py
#
# Local 2D map for MIRTE using ArUco markers as reference poses for
# rectangular physical boxes.
#
# Map coordinates:
#   x > 0 : right
#   z > 0 : forward
#
# Assumptions:
#   - Each ArUco marker is mounted flat on the FRONT face of its box.
#   - The marker is horizontally centered on that face.
#   - The visible/front face points toward MIRTE, so the box interior is
#     chosen as the marker-normal direction with positive map-z.
#
# Camera calibration is still approximate: distortion is assumed zero.

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../Mirte/ku_mirte_python"
    )
)

from ku_mirte import KU_Mirte


# ---------------------------------------------------------------------------
# Camera / ArUco calibration
# ---------------------------------------------------------------------------

ARUCO_MARKER_LENGTH_MM = 145.0

F = 609.9
CX = 640.0 / 2.0
CY = 480.0 / 2.0

INTRINSIC_MATRIX = np.array([
    [F,   0.0, CX],
    [0.0, F,   CY],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DISTORTION_COEFFS = np.zeros((5, 1), dtype=np.float64)

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)


# ---------------------------------------------------------------------------
# Physical geometry
# ---------------------------------------------------------------------------

# Measure these on the real setup and update if necessary.
CAMERA_FORWARD_OFFSET_M = 0.14
DEFAULT_BOX_WIDTH_M = 0.40
DEFAULT_BOX_DEPTH_M = 0.25

CAMERA_OFFSET_FROM_ROBOT_CENTER = np.array(
    [0.0, CAMERA_FORWARD_OFFSET_M],
    dtype=float
)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def marker_pose_to_box(
    tvec,
    rvec,
    camera_offset,
    width=DEFAULT_BOX_WIDTH_M,
    depth=DEFAULT_BOX_DEPTH_M
):
    """
    Convert one ArUco pose to an oriented rectangular box in the x-z map.

    tvec is the ArUco center relative to the camera.
    rvec is the ArUco orientation relative to the camera.

    The ArUco marker is assumed to be centered on the FRONT face.
    """
    tvec = np.asarray(tvec, dtype=float).reshape(3)
    rvec = np.asarray(rvec, dtype=float).reshape(3)

    marker_camera_m = tvec / 1000.0
    rotation, _ = cv2.Rodrigues(rvec)

    # Marker-local +Z axis expressed in camera coordinates.
    marker_normal_3d = rotation[:, 2]

    # Project marker normal to the floor plane (camera x-z).
    normal_xz = np.array(
        [marker_normal_3d[0], marker_normal_3d[2]],
        dtype=float
    )

    if np.linalg.norm(normal_xz) < 1e-6:
        raise ValueError(
            "ArUco orientation is degenerate in the x-z plane."
        )

    depth_direction = _unit(normal_xz)

    # For this exercise the marker is on the front face visible from MIRTE.
    # Therefore the box interior must point generally away from the camera.
    if depth_direction[1] < 0.0:
        depth_direction = -depth_direction

    # Perpendicular unit vector along box width.
    width_direction = np.array(
        [depth_direction[1], -depth_direction[0]],
        dtype=float
    )

    marker_position = np.array(
        [marker_camera_m[0], marker_camera_m[2]],
        dtype=float
    ) + np.asarray(camera_offset, dtype=float)

    box_center = (
        marker_position
        + depth_direction * (depth / 2.0)
    )

    half_width = width / 2.0
    half_depth = depth / 2.0

    front_center = box_center - depth_direction * half_depth
    back_center = box_center + depth_direction * half_depth

    corners = np.array([
        front_center - width_direction * half_width,
        front_center + width_direction * half_width,
        back_center + width_direction * half_width,
        back_center - width_direction * half_width
    ])

    theta_rad = float(
        np.arctan2(depth_direction[0], depth_direction[1])
    )

    return {
        "marker_position": marker_position,
        "box_center": box_center,
        "depth_direction": depth_direction,
        "width_direction": width_direction,
        "corners": corners,
        "theta_rad": theta_rad,
        "theta_deg": float(np.degrees(theta_rad)),
        "width": float(width),
        "depth": float(depth),
        "rvec": rvec.copy(),
        "tvec_mm": tvec.copy(),
        "rotation_matrix": rotation
    }


class LocalMap:
    """
    Authoritative local geometry model.

    self.boxes:
        Dictionary indexed by ArUco ID. Each entry contains the complete
        oriented rectangle geometry.

    self.landmarks:
        Legacy compatibility interface:
            [[box_center, marker_id], ...]
        New code should prefer get_box()/get_box_*().
    """

    def __init__(
        self,
        landmarks=None,
        mirte_radius=0.22,
        camera_offset=CAMERA_OFFSET_FROM_ROBOT_CENTER,
        low=(-2.0, 0.0),
        high=(2.0, 3.0),
        res=0.10,
        box_dimensions=None
    ):
        self.landmarks = [] if landmarks is None else landmarks
        self.boxes = {}

        self.mirte_radius = float(mirte_radius)
        self.camera_offset = np.asarray(camera_offset, dtype=float)

        self.map_area = [
            np.asarray(low, dtype=float),
            np.asarray(high, dtype=float)
        ]
        self.resolution = float(res)

        # Optional per-ID dimensions:
        # {7: (width, depth), 3: (width, depth)}
        self.box_dimensions = {} if box_dimensions is None else dict(box_dimensions)

    def _dimensions_for_id(self, marker_id):
        return self.box_dimensions.get(
            int(marker_id),
            (DEFAULT_BOX_WIDTH_M, DEFAULT_BOX_DEPTH_M)
        )

    def update(self, mirte):
        self.landmarks = self.get_map_from_mirte(mirte)

    # -----------------------------------------------------------------------
    # Public geometry API
    # -----------------------------------------------------------------------

    def get_visible_box_ids(self):
        return sorted(self.boxes.keys())

    def get_box(self, marker_id):
        return self.boxes.get(int(marker_id))

    def get_box_center(self, marker_id):
        box = self.get_box(marker_id)
        return None if box is None else box["box_center"].copy()

    def get_box_corners(self, marker_id):
        box = self.get_box(marker_id)
        return None if box is None else box["corners"].copy()

    def get_marker_position(self, marker_id):
        box = self.get_box(marker_id)
        return None if box is None else box["marker_position"].copy()

    def distance_to_box(self, point, marker_id):
        """
        Shortest Euclidean distance from a point to the physical rectangle.
        Returns 0 for a point inside the box.
        """
        box = self.get_box(marker_id)
        if box is None:
            return None

        point = np.asarray(point, dtype=float)
        rel = point - box["box_center"]

        local_width = abs(np.dot(rel, box["width_direction"]))
        local_depth = abs(np.dot(rel, box["depth_direction"]))

        dx = max(local_width - box["width"] / 2.0, 0.0)
        dz = max(local_depth - box["depth"] / 2.0, 0.0)

        return float(np.hypot(dx, dz))

    def in_collision(self, pos):
        """
        MIRTE is approximated as a circle and boxes as oriented rectangles.
        """
        pos = np.asarray(pos, dtype=float)

        if np.any(pos < self.map_area[0]) or np.any(pos > self.map_area[1]):
            return 1

        for marker_id in self.get_visible_box_ids():
            distance = self.distance_to_box(pos, marker_id)
            if distance is not None and distance <= self.mirte_radius:
                return 1

        return 0

    # -----------------------------------------------------------------------
    # Perception
    # -----------------------------------------------------------------------

    def get_map_from_mirte(self, mirte):
        landmark_map = []
        self.boxes = {}

        img = mirte.get_image_compressed()
        if img is None:
            print("No camera image received.")
            return landmark_map

        corners, ids, _ = cv2.aruco.detectMarkers(
            img,
            ARUCO_DICT
        )

        if ids is None:
            print("No ArUco landmarks detected.")
            return landmark_map

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            ARUCO_MARKER_LENGTH_MM,
            INTRINSIC_MATRIX,
            DISTORTION_COEFFS
        )

        for i, marker_array in enumerate(ids):
            marker_id = int(marker_array[0])

            # Ignore duplicate detections of an already stored ID.
            if marker_id in self.boxes:
                continue

            width, depth = self._dimensions_for_id(marker_id)

            try:
                box = marker_pose_to_box(
                    tvecs[i][0],
                    rvecs[i][0],
                    self.camera_offset,
                    width=width,
                    depth=depth
                )
            except ValueError as error:
                print(f"Skipping ID {marker_id}: {error}")
                continue

            box["id"] = marker_id
            self.boxes[marker_id] = box

            marker = box["marker_position"]
            center = box["box_center"]

            print(
                f"ID {marker_id:3d}: "
                f"ArUco=({marker[0]:+.3f}, {marker[1]:+.3f}) m | "
                f"box center=({center[0]:+.3f}, {center[1]:+.3f}) m | "
                f"orientation={box['theta_deg']:+.1f} deg | "
                f"size={box['width']:.3f} x {box['depth']:.3f} m"
            )

            # Legacy interface for existing project files.
            landmark_map.append([
                center.copy(),
                marker_id
            ])

        return landmark_map

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

    def draw_map(self):
        ax = plt.gca()

        for marker_id in self.get_visible_box_ids():
            box = self.boxes[marker_id]
            corners = box["corners"]
            center = box["box_center"]
            marker = box["marker_position"]
            direction = box["depth_direction"]

            ax.add_patch(
                Polygon(
                    corners,
                    closed=True,
                    fill=False
                )
            )

            ax.scatter(center[0], center[1])
            ax.scatter(marker[0], marker[1], marker="x")

            ax.arrow(
                center[0],
                center[1],
                direction[0] * 0.15,
                direction[1] * 0.15,
                head_width=0.03,
                length_includes_head=True
            )

            ax.text(
                center[0] + 0.02,
                center[1] + 0.02,
                f"ID {marker_id}  {box['theta_deg']:+.1f} deg"
            )

        ax.scatter(0.0, 0.0, marker="^")
        ax.set_xlim(self.map_area[0][0], self.map_area[1][0])
        ax.set_ylim(self.map_area[0][1], self.map_area[1][1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
