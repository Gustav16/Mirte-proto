from aruco_compat import detect as detect_markers, estimate as estimate_markers
"""ArUco observations and box geometry for Exercise 5.

Coordinate convention used by the Exercise-5 solution:
    x > 0      : right
    z > 0      : forward
    theta = 0  : facing +z
    positive theta / bearing : left (towards -x)

Two different points must not be mixed:

1. ArUco marker centre
   Used by MCL. Exercise 5 observes distance/orientation to the landmark via
   the ArUco code, so the fixed landmark map also describes ArUco centres.

2. Physical box centre
   Useful for plotting/collision/debugging. This uses the Exercise-4 geometry
   correction with NEGATIVE marker-local Z:

       box_center_camera = tvec + R @ [0, 0, -BOX_DEPTH/2]

The camera-forward offset is then added so both are expressed relative to the
ROBOT CENTRE.
"""

import hashlib
import weakref
import cv2
import numpy as np

from ex5_config import (
    ARUCO_MARKER_LENGTH_MM,
    BOX_CENTER_LOCAL_Z_OFFSET_MM,
    BOX_DEPTH_M,
    BOX_WIDTH_M,
    CAMERA_FORWARD_OFFSET_M,
    FOCAL_LENGTH_PX,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
INTRINSIC_MATRIX = np.array(
    [
        [FOCAL_LENGTH_PX, 0.0, IMAGE_WIDTH / 2.0],
        [0.0, FOCAL_LENGTH_PX, IMAGE_HEIGHT / 2.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DISTORTION_COEFFS = np.zeros((5, 1), dtype=np.float64)
CAMERA_OFFSET_XZ_M = np.array([0.0, CAMERA_FORWARD_OFFSET_M], dtype=float)


def pose_to_robot_geometry(tvec_mm, rvec):
    """Convert one OpenCV ArUco pose to robot-centred marker/box geometry.

    This is intentionally the same box-centre calculation as the existing
    Exercise-4 implementation.
    """
    tvec_mm = np.asarray(tvec_mm, dtype=float).reshape(3)
    rvec = np.asarray(rvec, dtype=float).reshape(3)

    rotation, _ = cv2.Rodrigues(rvec)

    # ArUco centre in robot-centre x-z coordinates.
    marker_camera_xz_m = np.array(
        [tvec_mm[0] / 1000.0, tvec_mm[2] / 1000.0],
        dtype=float,
    )
    marker_robot_xz_m = marker_camera_xz_m + CAMERA_OFFSET_XZ_M

    # IMPORTANT: the minus sign is deliberate and comes from the Ex4 tests.
    # Offset is expressed in the MARKER coordinate frame, then rotated into
    # the camera frame before extracting x/z.
    box_center_offset_local_mm = np.array(
        [0.0, 0.0, BOX_CENTER_LOCAL_Z_OFFSET_MM],
        dtype=float,
    )
    box_offset_camera_mm = rotation @ box_center_offset_local_mm
    box_offset_xz_m = np.array(
        [box_offset_camera_mm[0] / 1000.0, box_offset_camera_mm[2] / 1000.0],
        dtype=float,
    )
    box_center_robot_xz_m = marker_robot_xz_m + box_offset_xz_m

    # Unit direction from marker plane towards the inferred box centre.
    norm = np.linalg.norm(box_offset_xz_m)
    if norm > 1e-9:
        box_depth_direction = box_offset_xz_m / norm
    else:
        box_depth_direction = np.array([0.0, -1.0], dtype=float)

    # Width direction in the floor plane.
    box_width_direction = np.array(
        [box_depth_direction[1], -box_depth_direction[0]],
        dtype=float,
    )

    if not np.all(np.isfinite(tvec_mm)) or not np.all(np.isfinite(rvec)) or tvec_mm[2] <= 0:
        raise ValueError("Invalid ArUco pose.")
    return {
        "marker_position": marker_robot_xz_m,
        "box_center": box_center_robot_xz_m,
        "box_depth_direction": box_depth_direction,
        "box_width_direction": box_width_direction,
        "box_width": float(BOX_WIDTH_M),
        "box_depth": float(BOX_DEPTH_M),
        "rotation_matrix": rotation,
        "tvec_mm": tvec_mm.copy(),
        "rvec": rvec.copy(),
    }


def detect_geometry_from_image(image, allowed_ids=None):
    """Return per-ID marker and physical-box geometry from one image."""
    if image is None:
        return []
    if image.shape[:2] != (IMAGE_HEIGHT,IMAGE_WIDTH):
        raise ValueError("Camera resolution does not match calibration.")

    corners, ids, _ = detect_markers(image, ARUCO_DICT)
    if ids is None or len(ids) == 0:
        return []

    rvecs, tvecs, _ = estimate_markers(
        corners,
        ARUCO_MARKER_LENGTH_MM,
        INTRINSIC_MATRIX,
        DISTORTION_COEFFS,
    )

    allowed = None if allowed_ids is None else {int(v) for v in allowed_ids}
    result = []
    seen = set()

    for idx, marker_arr in enumerate(ids):
        marker_id = int(marker_arr[0])
        if marker_id in seen:
            continue
        if allowed is not None and marker_id not in allowed:
            continue
        seen.add(marker_id)

        geometry = pose_to_robot_geometry(tvecs[idx][0], rvecs[idx][0])
        geometry["id"] = marker_id
        result.append(geometry)

    return result


def detect_observations_from_image(image, allowed_ids=None):
    """Return [(id, range_m, bearing_rad), ...] for MCL.

    MCL deliberately uses the ARUCO MARKER CENTRE, not the physical box centre.
    The negative box-depth correction is therefore NOT applied to the MCL
    measurement itself. It is only used when physical box geometry is needed.
    """
    result = []

    for geometry in detect_geometry_from_image(image, allowed_ids=allowed_ids):
        x_m, z_m = geometry["marker_position"]
        range_m = float(np.hypot(x_m, z_m))
        bearing_rad = float(np.arctan2(-x_m, z_m))
        result.append((int(geometry["id"]), range_m, bearing_rad))

    return result


_last_frame_digests = weakref.WeakKeyDictionary()

def observe_mirte(mirte, allowed_ids=None):
    image = mirte.get_image_compressed()
    if image is None: return []
    # KU_Mirte exposes no image timestamp here. Reject byte-identical images
    # to avoid reusing a frozen frame as fresh independent evidence.
    digest=hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()
    try:
        old=_last_frame_digests.get(mirte)
        _last_frame_digests[mirte]=digest
    except TypeError:
        old=getattr(mirte,"_last_aruco_frame_digest",None)
        setattr(mirte,"_last_aruco_frame_digest",digest)
    if old==digest: return []
    return detect_observations_from_image(image, allowed_ids=allowed_ids)


def observe_geometry_mirte(mirte, allowed_ids=None):
    image = mirte.get_image_compressed()
    return detect_geometry_from_image(image, allowed_ids=allowed_ids)
