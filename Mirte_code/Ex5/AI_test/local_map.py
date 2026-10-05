from aruco_compat import detect as detect_markers, estimate as estimate_markers
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
#     chosen as marker-local negative Z, rotated into the camera frame.
#
# Camera calibration is still approximate: distortion is assumed zero.

import copy
from geometry_utils import segment_rectangle_distance
import ex5_config as cfg
from ex5_config import BOX_CLEARANCE_MARGIN_M
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

from robot_io import KU_Mirte


# ---------------------------------------------------------------------------
# Camera / ArUco calibration
# ---------------------------------------------------------------------------

ARUCO_MARKER_LENGTH_MM = cfg.ARUCO_MARKER_LENGTH_MM

F = cfg.FOCAL_LENGTH_PX
CX = cfg.IMAGE_WIDTH / 2.0
CY = cfg.IMAGE_HEIGHT / 2.0

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
CAMERA_FORWARD_OFFSET_M = cfg.CAMERA_FORWARD_OFFSET_M
DEFAULT_BOX_WIDTH_M = cfg.BOX_WIDTH_M
DEFAULT_BOX_DEPTH_M = cfg.BOX_DEPTH_M

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
    """Convert one ArUco pose to an oriented physical box in the x-z map.

    IMPORTANT: this matches the existing Ex4 convention. The physical box
    centre is reached from the ArUco plane with marker-local NEGATIVE Z:

        center_camera_mm = tvec + R @ [0, 0, -depth_mm/2]

    The previous AI-test version forced the box to marker +Z / positive map-z,
    which is the sign that placed the box incorrectly in your physical setup.
    """
    tvec = np.asarray(tvec, dtype=float).reshape(3)
    rvec = np.asarray(rvec, dtype=float).reshape(3)

    if width <= 0 or depth <= 0 or not np.all(np.isfinite([width,depth])):
        raise ValueError("Positive finite box dimensions required.")
    if not np.all(np.isfinite(tvec)) or not np.all(np.isfinite(rvec)) or tvec[2] <= 0:
        raise ValueError("Invalid marker pose.")
    rotation, _ = cv2.Rodrigues(rvec)

    marker_camera_m = tvec / 1000.0
    marker_position = np.array(
        [marker_camera_m[0], marker_camera_m[2]],
        dtype=float
    ) + np.asarray(camera_offset, dtype=float)

    # Same negative offset as current Ex4 local_map.py.
    obstacle_offset_3d_mm = np.array([
        0.0,
        0.0,
        -(float(depth) * 1000.0) / 2.0,
    ])
    box_offset_camera_mm = rotation @ obstacle_offset_3d_mm
    box_offset_xz = np.array([
        box_offset_camera_mm[0] / 1000.0,
        box_offset_camera_mm[2] / 1000.0,
    ])

    box_center = marker_position + box_offset_xz

    if np.linalg.norm(box_offset_xz) < 1e-9:
        raise ValueError("ArUco orientation is degenerate in the x-z plane.")

    depth_direction = _unit(box_offset_xz)
    width_direction = np.array(
        [depth_direction[1], -depth_direction[0]],
        dtype=float
    )

    half_width = width / 2.0
    half_depth = depth / 2.0

    # Marker is at the centre of the front face under this convention.
    front_center = box_center - depth_direction * half_depth
    back_center = box_center + depth_direction * half_depth

    corners = np.array([
        front_center - width_direction * half_width,
        front_center + width_direction * half_width,
        back_center + width_direction * half_width,
        back_center - width_direction * half_width
    ])

    theta_rad = float(np.arctan2(depth_direction[0], depth_direction[1]))

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
        "rotation_matrix": rotation,
        "box_offset_camera_mm": box_offset_camera_mm.copy(),
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
        clearance_margin=BOX_CLEARANCE_MARGIN_M,
        landmark_radius=0.20,
        mirte_radius=0.22,
        camera_offset=CAMERA_OFFSET_FROM_ROBOT_CENTER,
        low=(-2.0, 0.0),
        high=(2.0, 3.0),
        res=0.10,
        box_dimensions=None
    ):
        if landmarks:
            raise ValueError("Point-only obstacles lack box geometry. Use set_boxes with complete rectangles.")
        self.landmarks = []
        self.boxes = {}
        self.clearance_margin = float(clearance_margin)
        if self.clearance_margin < 0: raise ValueError("Negative clearance margin")

        self.landmark_radius = float(landmark_radius)
        self.mirte_radius = float(mirte_radius)
        self.camera_offset = np.asarray(camera_offset, dtype=float)

        self.map_area = [
            np.asarray(low, dtype=float),
            np.asarray(high, dtype=float)
        ]
        self.resolution = float(res)
        self.extent = [float(low[0]), float(high[0]), float(low[1]), float(high[1])]

        # Optional per-ID dimensions:
        # {7: (width, depth), 3: (width, depth)}
        self.box_dimensions = {} if box_dimensions is None else dict(box_dimensions)

    def _dimensions_for_id(self, marker_id):
        return self.box_dimensions.get(
            int(marker_id),
            (DEFAULT_BOX_WIDTH_M, DEFAULT_BOX_DEPTH_M)
        )

    def update(self, mirte):
        self.get_map_from_mirte(mirte)

    # -----------------------------------------------------------------------
    # Public geometry API
    # -----------------------------------------------------------------------

    def set_boxes(self, boxes):
        self.boxes = copy.deepcopy(boxes)
        self.landmarks = [[box["box_center"].copy(), int(i)] for i,box in sorted(self.boxes.items())]

    def segment_is_free(self, a, b):
        a,b=np.asarray(a,float),np.asarray(b,float)
        if a.shape!=(2,) or b.shape!=(2,) or not np.all(np.isfinite([a,b])): return False
        if np.any(a<self.map_area[0]) or np.any(a>self.map_area[1]) or np.any(b<self.map_area[0]) or np.any(b>self.map_area[1]): return False
        limit=self.mirte_radius+self.clearance_margin
        return all(segment_rectangle_distance(a,b,box)>limit for box in self.boxes.values())

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
        if pos.shape != (2,) or not np.all(np.isfinite(pos)): return 1

        if np.any(pos < self.map_area[0]) or np.any(pos > self.map_area[1]):
            return 1

        for marker_id in self.get_visible_box_ids():
            distance = self.distance_to_box(pos, marker_id)
            if distance is not None and distance <= self.mirte_radius + self.clearance_margin:
                return 1

        return 0

    # -----------------------------------------------------------------------
    # Perception
    # -----------------------------------------------------------------------

    def get_map_from_mirte(self, mirte):
        landmark_map = []
        self.set_boxes({})

        img = mirte.get_image_compressed()
        if img is not None and (img.shape[:2] != (cfg.IMAGE_HEIGHT,cfg.IMAGE_WIDTH)):
            raise ValueError("Camera image must match the calibrated 640x480 resolution.")
        if img is None:
            print("No camera image received.")
            return landmark_map

        corners, ids, _ = detect_markers(
            img,
            ARUCO_DICT
        )

        if ids is None:
            print("No ArUco landmarks detected.")
            return landmark_map

        rvecs, tvecs, _ = estimate_markers(
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

        self.set_boxes(self.boxes)
        return self.landmarks

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
