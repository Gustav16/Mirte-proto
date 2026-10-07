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

LANDMARK_ID_A = 4
LANDMARK_ID_B = 2
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
CAMERA_FX_PX = FOCAL_LENGTH_PX
CAMERA_FY_PX = FOCAL_LENGTH_PX
CAMERA_CX_PX = IMAGE_WIDTH / 2.0
CAMERA_CY_PX = IMAGE_HEIGHT / 2.0
# Zero is an assumption from the old scripts, NOT measured distortion.
CAMERA_DISTORTION_COEFFS = (0., 0., 0., 0., 0.)

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

# Continuous driving: update velocities, never stop between normal control ticks.
SMOOTH_LINEAR_SPEED_M_S = 0.06
SMOOTH_ANGULAR_SPEED_RAD_S = 0.40
SMOOTH_CONTROL_PERIOD_S = 0.15
SMOOTH_ACCELERATION_M_S2 = 0.10
SMOOTH_ANGULAR_ACCELERATION_RAD_S2 = 0.65
SMOOTH_COLLISION_HORIZON_S = 1.0
SMOOTH_MAX_TICKS = 2400
SMOOTH_MAX_TIME_S = 300.0
SMOOTH_SCAN_SPEED_RAD_S = 0.30
SMOOTH_SCAN_MAX_TIME_S = 40.0
SMOOTH_SCAN_HOLD_FRAMES = 16
SMOOTH_MIN_VIEW_SPEED_M_S = 0.008
SMOOTH_MAX_BLIND_DISTANCE_M = 0.24
SMOOTH_VIEW_MAX_TIME_S = 20.0

# Exercise 5 assumes both front markers face -z and boxes extend into +z.
# Set these world-frame marker-to-box-centre offsets to the actual setup.
LANDMARK_BOX_CENTER_OFFSETS_M = {i:(0.,BOX_DEPTH_M/2) for i in LANDMARKS}

# Driver settings used in ContinuousDrive.py and the later Ex1/2 scripts.
# These configure the driver, not MCL's physical velocity. Do not multiply
# MCL velocities by these values. ku_mirte.py is still unavailable for review.
DRIVE_SPEED_MODIFIER = 2.38
DRIVE_TURN_MODIFIER = 2.38

# Physical response calibration AFTER setting the above modifiers:
# actual_v = DRIVE_LINEAR_GAIN * command_v
# actual_w = directional_gain * command_w + DRIVE_DRIFT_RAD_PER_M * actual_v
# Neutral values until measurements exist in this SAME driver/speed regime.
DRIVE_LINEAR_GAIN = 1.0
DRIVE_LEFT_GAIN = 1.0
DRIVE_RIGHT_GAIN = 1.0
DRIVE_DRIFT_RAD_PER_M = 0.0

# Old comments describe a forward dead zone near 0.12 and spinning at 1.5.
# They do not prove stable low-speed operation. Verify the .008-.06 forward
# and .30 scan settings physically; never clamp them secretly in the driver.
DRIVE_LOW_SPEED_VERIFIED = False

# Adapted from the colleague's augmented MCL. Experimental and opt-in:
# it does not authorize driving with an uncertain/recovering pose.
AUGMENTED_MCL_ENABLED = False
AUGMENTED_MCL_FAST_ALPHA = 0.10
AUGMENTED_MCL_SLOW_ALPHA = 0.001
AUGMENTED_MCL_MAX_RANDOM_FRACTION = 0.10
