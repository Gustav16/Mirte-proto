import math

# ---------------------------------------------------------------------------
# Landmark world frame (Exercise 5)
# ---------------------------------------------------------------------------
# IMPORTANT: LANDMARK_DISTANCE_M must be measured between the TWO ArUco marker
# centres, because the Exercise-5 camera observation is range/bearing to the
# ArUco marker itself. Do not measure between box centres or box edges here.
#
# World frame:
#   ID_A = (0, 0)
#   ID_B = (LANDMARK_DISTANCE_M, 0)
#   theta = 0 means facing +z

LANDMARK_ID_A = 1
LANDMARK_ID_B = 10
LANDMARK_DISTANCE_M = 1.20  # REPLACE with the measured ArUco-centre distance.

LANDMARKS = {
    LANDMARK_ID_A: (0.0, 0.0),
    LANDMARK_ID_B: (LANDMARK_DISTANCE_M, 0.0),
}

GOAL = (
    LANDMARK_DISTANCE_M / 2.0,
    0.0,
)

# ---------------------------------------------------------------------------
# Camera / ArUco calibration - copied from the existing Exercise 4 setup
# ---------------------------------------------------------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOCAL_LENGTH_PX = 609.9
ARUCO_MARKER_LENGTH_MM = 145.0

# Camera is mounted 14 cm in front of the MIRTE centre. Ex4 established that
# this offset is ADDED when converting a camera-frame landmark position to the
# robot-centre frame.
CAMERA_FORWARD_OFFSET_M = 0.14

# ---------------------------------------------------------------------------
# Physical box geometry - same convention as the existing Exercise 4 code
# ---------------------------------------------------------------------------
# Ex4 models the obstacle/box as about 25 cm deep. The crucial existing
# sign is NEGATIVE local marker Z:
#
#     box_center_camera = tvec + R @ [0, 0, -BOX_DEPTH/2]
#
# Do not change this to +BOX_DEPTH/2. That was the version which placed the
# physical box on the wrong side of the ArUco plane in your setup.
BOX_DEPTH_M = 0.25
BOX_DEPTH_MM = BOX_DEPTH_M * 1000.0
BOX_CENTER_LOCAL_Z_OFFSET_MM = -BOX_DEPTH_MM / 2.0

# Ex4 used a 0.20 m landmark radius and a 0.22 m MIRTE radius for collision
# checking. The old AI-test rectangle used width 0.40 m, consistent with a
# 0.20 m half-width.
LANDMARK_RADIUS_M = 0.20
BOX_WIDTH_M = 2.0 * LANDMARK_RADIUS_M
MIRTE_RADIUS_M = 0.22

# ---------------------------------------------------------------------------
# MCL parameters
# ---------------------------------------------------------------------------
NUMBER_OF_PARTICLES = 3000

# Global initial search area in the landmark frame.
INITIAL_X_MIN = -1.0
INITIAL_X_MAX = LANDMARK_DISTANCE_M + 1.0
INITIAL_Z_MIN = -2.0
INITIAL_Z_MAX = 2.0

SIGMA_RANGE_M = 0.10
SIGMA_BEARING_RAD = math.radians(8.0)

ALPHA_TRANSLATION = 0.08
ALPHA_ROTATION = 0.08
TRANSLATION_NOISE_FLOOR_M = 0.005
ROTATION_NOISE_FLOOR_RAD = math.radians(0.5)

ROUGHEN_POSITION_M = 0.003
ROUGHEN_ANGLE_RAD = math.radians(0.3)

# ---------------------------------------------------------------------------
# Robot control
# ---------------------------------------------------------------------------
LINEAR_SPEED = 0.22
ANGULAR_SPEED = 0.65
MAX_TRANSLATION_STEP_M = 0.12
MAX_ROTATION_STEP_RAD = math.radians(15.0)
HEADING_TOLERANCE_RAD = math.radians(8.0)
GOAL_TOLERANCE_M = 0.10
MAX_CONTROL_STEPS = 80
STRAIGHT_ANGULAR_BIAS = 0.0
SONAR_STOP_DISTANCE_M = 0.25

# Closed-loop observation/recovery limits. Speeds remain the original values.
CAMERA_SETTLE_SECONDS = 0.15
LOCALIZATION_FRAMES = 16
MAX_SCAN_STEPS = 24  # 24 x 15 degrees = one revolution, no translation.
LOCALIZATION_POSITION_STD_M = 0.06
LOCALIZATION_ANGLE_STD_RAD = math.radians(12)
RESCAN_TRANSLATION_M = 0.24
GOAL_SCAN_DISTANCE_M = 0.22
PATH_MAX_UNOBSERVED_DISTANCE_M = 0.24
BOX_CLEARANCE_MARGIN_M = 0.025

FINAL_MAX_UNOBSERVED_DISTANCE_M = 0.24  # only after a near-goal scan of BOTH IDs

FINAL_APPROACH_STANDOFF_M = 0.005
