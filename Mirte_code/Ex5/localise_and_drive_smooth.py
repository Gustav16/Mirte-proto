"""
localise_and_drive_smooth.py

A copy of localise_and_drive.py with smooth driving: MIRTE finds out where it
is from two ArUco boxes, then drives to the midpoint between them in one
continuous motion and arrives facing through the gap.

    python3 localise_and_drive_smooth.py                       # on MIRTE, asks: drive or only measure?
    python3 localise_and_drive_smooth.py --measure             # only measure and plot: MIRTE does not move at all
    python3 localise_and_drive_smooth.py --drive               # drive without asking
    python3 localise_and_drive_smooth.py --sim --drive         # simulated MIRTE, no robot needed
    python3 localise_and_drive_smooth.py --sim 0.8 -2.2 60 --drive   # simulated start pose: x (m), y (m), heading (deg)

Steps
  1. Scan: MIRTE turns on the spot and looks. The first two ArUco boxes the
     camera finds become the landmarks; no ids or positions are typed in.
     Each box gets its own set of particles. A particle is one guess of where
     that box is in MIRTE's own frame. Every camera reading re-weights and
     resamples the guesses, and every movement of MIRTE moves them the
     opposite way, with noise. Together the box particles are the local map.
  2. Align: the global map is built from the two boxes: the one with the
     lowest id is the origin and the other lies on the +x axis. Then find the
     rotation and shift that lays the local map on top of the global map.
     That rotation and shift is MIRTE's pose.
  3. Plan: a Dubins path from MIRTE's pose to the goal pose. The goal pose is
     the midpoint between the boxes, facing straight through the gap. A Dubins
     path is made of arcs of radius TURN_RADIUS and one straight line, so MIRTE
     can turn while it drives and never has to stop to rotate. The shortest of
     the six Dubins paths that keeps clear of the boxes is used. If none is
     clear, the script falls back to RRT with stop-and-turn driving.
  4. Drive: the whole path is sent as drive commands one after the other,
     without stopping in between (non-holonomic: forward speed plus turning).

Measure only
  With "measure" MIRTE never moves or turns, so the experiment setup is left
  alone. It looks MEASURE_LOOKS times from where it stands and prints the mean,
  standard deviation, minimum and maximum of the distance and the bearing to
  each box. If both boxes are in view it also works out its pose and the path
  it would drive, and saves the usual plots. If not, it saves a plot of what
  it saw.
  5. Look again: scan once more to see where MIRTE ended up and which way it
     faced when it arrived.
  6. Plots: where MIRTE thought it was before, the planned path, and where it
     thought it was after moving. They are saved as one PNG.

Frames
  Robot frame:  x forward, y left, in metres. The local map lives here.
  Global frame: the frame of the two landmarks (see step 2). Heading 0 points
                along +x and grows to the left (counter-clockwise).
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import particle
import random_numbers as rn


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# The landmarks are found by the camera: the first NUM_LANDMARKS different ArUco
# boxes seen within MAX_LANDMARK_RANGE.
NUM_LANDMARKS = 2
MAX_LANDMARK_RANGE = 4.0        # m, markers further away are ignored (other boxes in the room)
LANDMARK_DISTANCE = None        # m between the two box centres if you have measured it.
                                # None = use the distance the camera measured.

# Global map: marker id -> (x, y) of the box centre in metres.
# Filled in by make_global_map() after the first scan.
LANDMARKS = {}

# The boxes of the simulated world (--sim): marker id -> (x, y) in metres
SIM_BOXES = {
    4: (0.0, 0.0),
    9: (3.0, 0.0)
}

BOX_PARTICLES = 200             # guesses per box
SIGMA_DIST = 0.10               # m, camera distance noise the filter assumes
SIGMA_ANGLE = 0.15              # rad, camera angle noise the filter assumes

CAMERA_OFFSET = 0.14            # m, camera in front of the robot centre (from Ex4)
BOX_DEPTH = 0.25                # m, the marker sits half a box in front of the box centre
BOX_RADIUS = 0.20               # m, as in Ex4's LocalMap
ROBOT_RADIUS = 0.22             # m, as in Ex4's LocalMap

MEASURE_LOOKS = 20              # looks taken from one spot in measure-only mode

TURN_STEP = math.radians(25)    # turn this much between looks when scanning (left)
MAX_TURNS = 15                  # 15 x 25 degrees is a little more than a full turn

LINEAR_SPEED = 0.3              # m/s,   driving calibration from Ex4's Execute_path
ANGULAR_SPEED = 0.7             # rad/s
LINEAR_OFFSET = -0.0354         # rad/s, keeps MIRTE straight while driving
TURN_SCALER_LEFT = 1.05
TURN_SCALER_RIGHT = 1.10

TURN_RADIUS = LINEAR_SPEED / ANGULAR_SPEED      # m, tightest arc at full speed (about 0.43 m)
PLAN_MARGIN = 0.10              # m, extra clearance to the boxes for the smooth path

RRT_STEP = 0.3                  # m
RRT_GOAL_BIAS = 0.1             # share of samples that are the goal itself
RRT_MAX_ITER = 2000

GOAL_TOLERANCE = 0.10           # m, close enough to the midpoint
MAX_ROUNDS = 1                  # 1 = one smooth drive. More = plan, drive and look again until close enough

POSE_SAMPLES = 300              # pose guesses drawn for the plots

ROBOT_AT_ORIGIN = particle.Particle(0.0, 0.0, 0.0)     # MIRTE in its own frame


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def wrap(angle):
    """Angle in (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def rotation(angle):
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle),  math.cos(angle)]])


def to_local(dist, angle):
    """Position in the robot frame of something the camera sees at dist (m) and angle (rad)."""
    return np.array([CAMERA_OFFSET + dist * math.cos(angle), dist * math.sin(angle)])


def to_global(pose, point):
    """A point in the robot frame, seen from the global frame, for a robot at pose (x, y, theta)."""
    return rotation(pose[2]) @ np.asarray(point) + np.array(pose[:2])


# ---------------------------------------------------------------------------
# The robot: the real MIRTE, or a simulated one with the same methods
# ---------------------------------------------------------------------------

class MirteRobot:
    def __init__(self):
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../Mirte/ku_mirte_python'))
        import cv2
        from ku_mirte import KU_Mirte
        self.cv2 = cv2
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        f = 609.9                                       # MIRTE camera calibration from Ex3
        self.intrinsic_matrix = np.array([[f, 0, 640 / 2], [0, f, 480 / 2], [0, 0, 1]], dtype=np.float32)
        self.mirte = KU_Mirte()
        time.sleep(1)       # wait for camera to setup

    def look(self):
        """{marker id: (distance m, angle rad)} from the camera to each box centre in view.
        The angle is from straight ahead, positive to the left."""
        # get_image_compressed() returns the last image that arrived, which may be old.
        # Wait for one that arrives after MIRTE has settled, so the image is fresh and not blurred.
        time.sleep(0.5)
        old = self.mirte.get_image_compressed()
        img = old
        waited = 0.0
        while img is old and waited < 3.0:
            time.sleep(0.05)
            waited += 0.05
            img = self.mirte.get_image_compressed()
        seen = {}
        if img is None or img is old:
            print("  no new camera image for 3 s (weak connection to MIRTE?)")
            return seen
        corners, ids, _ = self.cv2.aruco.detectMarkers(img, self.aruco_dict)
        if ids is None:
            return seen
        rvecs, tvecs, _ = self.cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.145, self.intrinsic_matrix, np.zeros(5))
        for i in range(len(ids)):
            x, y, z = tvecs[i][0]                       # camera frame: x right, y down, z forward
            dist = np.sqrt(x * x + z * z) + BOX_DEPTH / 2   # to the box centre, seen roughly face on
            marker_id = int(ids[i][0])
            if marker_id not in seen or dist < seen[marker_id][0]:
                seen[marker_id] = (dist, -np.arctan2(x, z))
        return seen

    def turn(self, angle):
        if abs(angle) < 1e-3:
            return
        scaler = TURN_SCALER_LEFT if angle > 0 else TURN_SCALER_RIGHT
        self.mirte.drive(0.0, math.copysign(ANGULAR_SPEED, angle), scaler * abs(angle) / ANGULAR_SPEED)

    def drive(self, dist):
        self.mirte.drive(LINEAR_SPEED, LINEAR_OFFSET, dist / LINEAR_SPEED)

    def follow(self, commands):
        """Run drive commands (linear speed, angular speed, seconds) one after the other
        without stopping in between. Each command is given a little extra time and is
        replaced by the next one before it runs out."""
        for lin, ang, duration in commands:
            scaler = 1.0 if ang == 0 else (TURN_SCALER_LEFT if ang > 0 else TURN_SCALER_RIGHT)
            start = time.time()
            self.mirte.drive(lin, ang * scaler + LINEAR_OFFSET, duration + 0.5, blocking=False)
            time.sleep(max(0.0, duration - (time.time() - start)))
        self.mirte.stop()

    def stop(self):
        self.mirte.stop()


class SimRobot:
    """A pretend MIRTE with a true pose in the global frame, noisy motion and a noisy camera."""
    HALF_FOV = math.atan(320 / 609.9)       # 27.7 degrees, from the camera calibration

    def __init__(self, x, y, heading):
        self.pose = particle.Particle(x, y, heading)
        self.track = [(x, y)]

    def in_map_frame(self, x, y, heading=0.0):
        """A pose of the simulated world, seen in the frame of the landmarks MIRTE found."""
        first, second = sorted(LANDMARKS)
        origin = np.array(SIM_BOXES[first])
        axis = np.array(SIM_BOXES[second]) - origin
        phi = math.atan2(axis[1], axis[0])
        px, py = rotation(-phi) @ (np.array([x, y]) - origin)
        return (px, py, wrap(heading - phi))

    def true_pose(self):
        return self.in_map_frame(self.pose.getX(), self.pose.getY(), self.pose.getTheta())

    def true_miss(self):
        """True distance (m) from the midpoint between the landmarks."""
        goal = np.mean([SIM_BOXES[i] for i in LANDMARKS], axis=0)
        return float(np.linalg.norm(np.array([self.pose.getX(), self.pose.getY()]) - goal))

    def look(self):
        seen = {}
        for marker_id, landmark in SIM_BOXES.items():
            dist, angle = particle.expected_measurement(self.pose, landmark, CAMERA_OFFSET)
            if abs(angle) <= self.HALF_FOV and 0.3 <= dist <= 8.0:
                seen[marker_id] = (dist * (1 + rn.randn(0.0, 0.015)) + rn.randn(0.0, 0.01),
                                   angle + rn.randn(0.0, math.radians(2)))
        return seen

    def turn(self, angle):
        if abs(angle) < 1e-3:
            return
        particle.move_particle(self.pose, 0.0, 0.0, angle * (1 + rn.randn(0.0, 0.07)) + rn.randn(0.0, math.radians(1)))

    def drive(self, dist):
        actual = dist * (1 + rn.randn(0.0, 0.05))
        drift = rn.randn(0.0, math.radians(2) * dist)
        heading = self.pose.getTheta() + drift / 2
        particle.move_particle(self.pose, actual * math.cos(heading), actual * math.sin(heading), drift)
        self.track.append((self.pose.getX(), self.pose.getY()))

    def follow(self, commands):
        speed_error = 1 + rn.randn(0.0, 0.05)           # the same for the whole drive
        turn_error = 1 + rn.randn(0.0, 0.07)
        drift = rn.randn(0.0, math.radians(2))          # rad per metre
        for lin, ang, duration in commands:
            steps = max(1, int(duration / 0.02))
            for k in range(steps):
                d = lin * speed_error * duration / steps
                heading = self.pose.getTheta()
                particle.move_particle(self.pose, d * math.cos(heading), d * math.sin(heading),
                                       ang * turn_error * duration / steps + drift * d)
                if k % 5 == 0:
                    self.track.append((self.pose.getX(), self.pose.getY()))
        self.track.append((self.pose.getX(), self.pose.getY()))

    def stop(self):
        pass


# ---------------------------------------------------------------------------
# Step 1: the local map. local_map = {marker id: list of particles}, where each
# particle is one guess of that box's position in the robot frame.
# ---------------------------------------------------------------------------

def look_and_update(robot, local_map):
    """Take one camera reading and update the particles of every landmark in view.
    Returns {marker id: (distance, angle)} for the landmarks seen.

    Markers further away than MAX_LANDMARK_RANGE are ignored. Until NUM_LANDMARKS boxes
    have been found, a new box becomes a landmark. After that, other boxes are ignored."""
    seen = {}
    for marker_id, (dist, angle) in sorted(robot.look().items(), key=lambda item: item[1][0]):    # nearest first
        if dist > MAX_LANDMARK_RANGE:
            print(f"    ignored box {marker_id} at {dist:.2f} m: further than MAX_LANDMARK_RANGE")
            continue
        if marker_id not in local_map and len(local_map) >= NUM_LANDMARKS:
            print(f"    ignored box {marker_id} at {dist:.2f} m: not one of the landmarks")
            continue
        seen[marker_id] = (dist, angle)
        if marker_id not in local_map:
            # first sighting: spread the guesses round where the camera says the box is
            local_map[marker_id] = []
            for k in range(BOX_PARTICLES):
                x, y = to_local(dist + rn.randn(0.0, SIGMA_DIST), angle + rn.randn(0.0, SIGMA_ANGLE))
                local_map[marker_id].append(particle.Particle(x, y, 0.0, 1.0 / BOX_PARTICLES))
        # weight each guess by how well it fits the reading, then normalise and resample
        for p in local_map[marker_id]:
            p.setWeight(particle.measurement_weight(
                ROBOT_AT_ORIGIN, dist, angle, (p.getX(), p.getY()), SIGMA_DIST, SIGMA_ANGLE, CAMERA_OFFSET))
        particle.normalise_weights(local_map[marker_id])
        local_map[marker_id] = particle.resample(local_map[marker_id])
        particle.add_uncertainty(local_map[marker_id], 0.01, 0.0)
    return seen


def move_local_map(local_map, turn=0.0, forward=0.0):
    """MIRTE turned (rad, left positive) and then drove forward (m): in its own frame
    the boxes move the opposite way. Each guess gets its own noise on the movement."""
    for particles in local_map.values():
        for p in particles:
            angle = -(turn + rn.randn(0.0, 0.07 * abs(turn) + 0.01)) if turn else 0.0
            angle -= rn.randn(0.0, 0.03 * forward) if forward else 0.0
            x, y = rotation(angle) @ np.array([p.getX(), p.getY()])
            p.setX(x - forward * (1 + rn.randn(0.0, 0.05)) if forward else x)
            p.setY(y)


def box_positions(local_map):
    """The local map as {marker id: average position of that box's particles}."""
    out = {}
    for marker_id, particles in local_map.items():
        est = particle.estimate_pose(particles)
        out[marker_id] = np.array([est.getX(), est.getY()])
    return out


def scan(robot, local_map):
    """Turn on the spot, looking after each turn, until every landmark has been seen in this scan.
    Returns (True if all were seen, how far MIRTE turned in total in rad)."""
    seen_now = set()
    turned = 0.0
    for step in range(MAX_TURNS):
        seen = look_and_update(robot, local_map)
        seen_now.update(seen)
        readings = ", ".join(f"box {i} at {d:.2f} m, {math.degrees(a):+.0f} deg" for i, (d, a) in sorted(seen.items()))
        print(f"  look {step + 1}: {readings if seen else 'no landmark in view'}")
        if len(seen_now) == NUM_LANDMARKS:
            return True, turned
        robot.turn(TURN_STEP)
        move_local_map(local_map, turn=TURN_STEP)
        turned += TURN_STEP
    return False, turned


def measure(robot, local_map):
    """Look MEASURE_LOOKS times without moving, and print how much the readings vary.
    Returns {marker id: list of (distance, angle)} with every reading taken."""
    readings = {}
    for k in range(MEASURE_LOOKS):
        for marker_id, reading in look_and_update(robot, local_map).items():
            readings.setdefault(marker_id, []).append(reading)
    if not readings:
        print("  no box in view")
    for marker_id, values in sorted(readings.items()):
        dist = np.array([v[0] for v in values])
        bearing = np.degrees([v[1] for v in values])
        print(f"  box {marker_id}: seen in {len(values)} of {MEASURE_LOOKS} looks")
        print(f"    distance: mean {dist.mean():.3f} m  std {dist.std():.3f} m  min {dist.min():.3f} m  max {dist.max():.3f} m")
        print(f"    bearing:  mean {bearing.mean():+.2f} deg  std {bearing.std():.2f} deg  "
              f"min {bearing.min():+.2f} deg  max {bearing.max():+.2f} deg")
    return readings


# ---------------------------------------------------------------------------
# Step 2: align the local map with the global map
# ---------------------------------------------------------------------------

def make_global_map(local_map):
    """Build the global map from the two boxes the camera found: the box with the lowest id
    is the origin and the other lies on the +x axis. Returns the measured distance between them."""
    boxes = box_positions(local_map)
    first, second = sorted(boxes)
    measured = float(np.linalg.norm(boxes[second] - boxes[first]))
    LANDMARKS.clear()
    LANDMARKS[first] = (0.0, 0.0)
    LANDMARKS[second] = (LANDMARK_DISTANCE if LANDMARK_DISTANCE is not None else measured, 0.0)
    return measured


def align(local, glob):
    """Least-squares fit of the local map onto the global map.
    local, glob: {marker id: (x, y)}. Returns MIRTE's pose (x, y, theta) in the global frame."""
    ids = sorted(set(local) & set(glob))
    L = np.array([local[i] for i in ids])
    G = np.array([glob[i] for i in ids])
    a = L - L.mean(axis=0)                  # both maps centred on their own middle
    b = G - G.mean(axis=0)
    theta = math.atan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    x, y = G.mean(axis=0) - rotation(theta) @ L.mean(axis=0)
    return (x, y, theta)


def sample_poses(local_map):
    """Many pose guesses: each one aligns one randomly picked particle per box.
    Their spread shows how sure MIRTE is of its pose."""
    poses = []
    for k in range(POSE_SAMPLES):
        local = {}
        for marker_id, particles in local_map.items():
            p = particles[np.random.randint(len(particles))]
            local[marker_id] = (p.getX(), p.getY())
        poses.append(align(local, LANDMARKS))
    return np.array(poses)


# ---------------------------------------------------------------------------
# Step 3: the smooth path (Dubins), in the robot frame
# ---------------------------------------------------------------------------

def goal_pose(boxes, heading=None):
    """The goal in the robot frame: the midpoint between the two boxes, facing straight
    through the gap, away from the side MIRTE is on. Returns (x, y, heading).
    If a heading (rad, robot frame) is given, it is used as it is."""
    first, second = sorted(boxes)
    middle = (boxes[first] + boxes[second]) / 2
    if heading is None:
        along = boxes[second] - boxes[first]
        through = np.array([-along[1], along[0]])           # square to the line between the boxes
        if through @ middle < 0:                            # point away from MIRTE, which is at the origin
            through = -through
        heading = math.atan2(through[1], through[0])
    return (middle[0], middle[1], heading)


def dubins_paths(goal, radius):
    """All Dubins paths from the origin, heading 0, to goal = (x, y, heading).

    A Dubins path has three parts, each a left arc (L), a right arc (R) or a straight
    line (S). Returns a list of (length in m, name, [(part, amount), ...]), shortest
    first, where amount is an angle in rad for an arc and a length in m for a line."""
    def mod2pi(a):
        return a % (2 * math.pi)

    d = math.hypot(goal[0], goal[1]) / radius
    theta = math.atan2(goal[1], goal[0])
    alpha, beta = mod2pi(-theta), mod2pi(goal[2] - theta)
    sa, sb, ca, cb, cab = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta), math.cos(alpha - beta)
    found = []

    p_sq = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
    if p_sq >= 0:
        tmp = math.atan2(cb - ca, d + sa - sb)
        found.append(("LSL", mod2pi(-alpha + tmp), math.sqrt(p_sq), mod2pi(beta - tmp)))
    p_sq = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
    if p_sq >= 0:
        tmp = math.atan2(ca - cb, d - sa + sb)
        found.append(("RSR", mod2pi(alpha - tmp), math.sqrt(p_sq), mod2pi(-beta + tmp)))
    p_sq = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
    if p_sq >= 0:
        p = math.sqrt(p_sq)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        found.append(("LSR", mod2pi(-alpha + tmp), p, mod2pi(-mod2pi(beta) + tmp)))
    p_sq = -2 + d * d + 2 * cab - 2 * d * (sa + sb)
    if p_sq >= 0:
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        found.append(("RSL", mod2pi(alpha - tmp), p, mod2pi(beta - tmp)))
    tmp = (6 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8
    if abs(tmp) <= 1:
        p = mod2pi(2 * math.pi - math.acos(tmp))
        t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2)
        found.append(("RLR", t, p, mod2pi(alpha - beta - t + p)))
    tmp = (6 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8
    if abs(tmp) <= 1:
        p = mod2pi(2 * math.pi - math.acos(tmp))
        t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2)
        found.append(("LRL", t, p, mod2pi(mod2pi(beta) - alpha - t + p)))

    paths = []
    for name, t, p, q in found:
        parts = [(kind, amount * radius if kind == "S" else amount) for kind, amount in zip(name, (t, p, q))]
        paths.append(((t + p + q) * radius, name, parts))
    return sorted(paths, key=lambda path: path[0])


def drive_commands(parts, radius):
    """Dubins parts as drive commands (linear speed m/s, angular speed rad/s, seconds)."""
    commands = []
    for kind, amount in parts:
        if amount < 1e-6:
            continue
        if kind == "S":
            commands.append((LINEAR_SPEED, 0.0, amount / LINEAR_SPEED))
        else:
            sign = 1.0 if kind == "L" else -1.0
            commands.append((LINEAR_SPEED, sign * LINEAR_SPEED / radius, amount * radius / LINEAR_SPEED))
    return commands


def path_points(commands):
    """The points MIRTE passes when it follows the commands exactly, from the origin, about 2 cm apart."""
    x, y, heading = 0.0, 0.0, 0.0
    points = [(x, y)]
    for lin, ang, duration in commands:
        steps = max(1, int(lin * duration / 0.02))
        dt = duration / steps
        for k in range(steps):
            heading += ang * dt / 2
            x += lin * dt * math.cos(heading)
            y += lin * dt * math.sin(heading)
            heading += ang * dt / 2
            points.append((x, y))
    return np.array(points)


def plan_smooth(boxes, goal):
    """The shortest Dubins path to the goal pose that keeps clear of the boxes.
    Returns a dict with name, length, commands and points, or None if no Dubins path is clear."""
    centres = np.array(list(boxes.values()))
    for margin in (PLAN_MARGIN, 0.0):
        for length, name, parts in dubins_paths(goal, TURN_RADIUS):
            commands = drive_commands(parts, TURN_RADIUS)
            points = path_points(commands)
            nearest = np.linalg.norm(points[:, None, :] - centres[None], axis=2).min()
            if nearest > BOX_RADIUS + ROBOT_RADIUS + margin:
                return dict(name=name, length=length, commands=commands, points=points,
                            seconds=sum(c[2] for c in commands), margin=margin)
    return None


# ---------------------------------------------------------------------------
# Fallback: RRT in the robot frame, driven with stop-and-turn
# ---------------------------------------------------------------------------

def plan_rrt(boxes, goal):
    """RRT from MIRTE (the origin) to the goal, round the boxes.
    Returns (tree edges, path, shortcut path). The paths are None if no path was found."""
    start = np.zeros(2)
    centres = np.array(list(boxes.values()))
    clearance = BOX_RADIUS + ROBOT_RADIUS
    everything = np.vstack([centres, start, goal])
    low, high = everything.min(axis=0) - 1.0, everything.max(axis=0) + 1.0

    def free(a, b):
        """True if the straight line a -> b stays out of every no-go zone."""
        n = max(2, int(np.linalg.norm(b - a) / 0.05) + 1)
        pts = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
        return bool(np.all(np.linalg.norm(pts[:, None, :] - centres[None], axis=2) > clearance))

    nodes, parents, edges, path = [start], [None], [], None
    for k in range(RRT_MAX_ITER):
        target = goal if np.random.ranf() < RRT_GOAL_BIAS else np.random.uniform(low, high)
        near = int(np.argmin([np.linalg.norm(n - target) for n in nodes]))
        d = np.linalg.norm(target - nodes[near])
        if d < 1e-9:
            continue
        new = target if d <= RRT_STEP else nodes[near] + (target - nodes[near]) * RRT_STEP / d
        if not free(nodes[near], new):
            continue
        nodes.append(new)
        parents.append(near)
        edges.append((nodes[near], new))
        if np.linalg.norm(new - goal) <= RRT_STEP and free(new, goal):
            path, i = [goal], len(nodes) - 1
            while i is not None:
                path.append(nodes[i])
                i = parents[i]
            path.reverse()
            break
    if path is None:
        return edges, None, None

    # shortcut: from each point jump to the furthest later point with a free straight line
    short, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not free(path[i], path[j]):
            j -= 1
        short.append(path[j])
        i = j
    return edges, path, short


# ---------------------------------------------------------------------------
# Step 4: drive the path
# ---------------------------------------------------------------------------

def move_local_map_along(local_map, commands):
    """MIRTE followed drive commands: move every box guess the opposite way.
    Each guess gets its own noise on the turning and on the distance of each command."""
    for lin, ang, duration in commands:
        for particles in local_map.values():
            for p in particles:
                turn = ang * duration * (1 + rn.randn(0.0, 0.07)) + rn.randn(0.0, 0.01)
                dist = lin * duration * (1 + rn.randn(0.0, 0.05))
                if abs(turn) < 1e-6:
                    moved = np.array([dist, 0.0])
                else:                                       # an arc of radius dist / turn
                    moved = dist / turn * np.array([math.sin(turn), 1 - math.cos(turn)])
                x, y = rotation(-turn) @ (np.array([p.getX(), p.getY()]) - moved)
                p.setX(x)
                p.setY(y)


def drive_path(robot, local_map, path):
    """Turn, then drive, for each leg. The path is in the robot frame MIRTE had when it planned."""
    position, heading = np.asarray(path[0]), 0.0
    for point in path[1:]:
        leg = np.asarray(point) - position
        dist = float(np.linalg.norm(leg))
        if dist < 1e-6:
            continue
        turn = wrap(math.atan2(leg[1], leg[0]) - heading)
        print(f"  turn {math.degrees(turn):+.0f} deg, drive {dist:.2f} m")

        robot.turn(turn)
        move_local_map(local_map, turn=turn)
        look_and_update(robot, local_map)

        robot.drive(dist)
        move_local_map(local_map, forward=dist)
        look_and_update(robot, local_map)

        position, heading = np.asarray(point), heading + turn


# ---------------------------------------------------------------------------
# Step 6: plots
# ---------------------------------------------------------------------------

def draw_world(ax, title):
    for marker_id, (x, y) in LANDMARKS.items():
        ax.add_patch(plt.Rectangle((x - BOX_DEPTH / 2, y - BOX_DEPTH / 2), BOX_DEPTH, BOX_DEPTH, color="0.25"))
        ax.text(x, y + 0.3, f"box {marker_id}", ha="center", fontsize=8)
    goal = np.mean(list(LANDMARKS.values()), axis=0)
    ax.plot(goal[0], goal[1], "x", color="green", markersize=11, markeredgewidth=2.5, label="midpoint (goal)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_title(title, fontsize=10)


def draw_pose(ax, pose, colour, label):
    ax.plot(pose[0], pose[1], "o", color=colour, markersize=7, label=label)
    ax.plot([pose[0], pose[0] + 0.35 * math.cos(pose[2])], [pose[1], pose[1] + 0.35 * math.sin(pose[2])],
            color=colour, linewidth=2)


def save_local_plot(local_map, readings, filename):
    """What MIRTE saw from where it stands, in its own frame. Used when it cannot work out its pose."""
    fig, ax = plt.subplots(figsize=(7, 7))
    half_fov = math.atan(320 / 609.9)
    reach = MAX_LANDMARK_RANGE
    ax.fill([CAMERA_OFFSET, CAMERA_OFFSET + reach, CAMERA_OFFSET + reach],
            [0.0, reach * math.tan(half_fov), -reach * math.tan(half_fov)], color="tab:blue", alpha=0.06,
            label="camera view")
    for marker_id, particles in sorted(local_map.items()):
        ax.scatter([p.getX() for p in particles], [p.getY() for p in particles], s=4, alpha=0.4,
                   label=f"box {marker_id}: particles")
        points = np.array([to_local(d, a) for d, a in readings.get(marker_id, [])])
        if len(points):
            ax.plot(points[:, 0], points[:, 1], "k+", markersize=6, label=f"box {marker_id}: camera readings")
    ax.add_patch(Circle((0.0, 0.0), ROBOT_RADIUS, color="magenta", alpha=0.3))
    ax.plot([0.0, 0.35], [0.0, 0.0], color="magenta", linewidth=2, label="MIRTE, facing right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x, forward (m)")
    ax.set_ylabel("y, left (m)")
    ax.set_title("Measure only: what MIRTE saw, in its own frame", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"Saved plot to: {filename}")


def save_plots(log, filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    goal = np.mean(list(LANDMARKS.values()), axis=0)

    # 1. before moving
    ax = axes[0]
    draw_world(ax, "1. Before moving: where MIRTE thought it was")
    ax.scatter(log["samples_before"][:, 0], log["samples_before"][:, 1], s=4, color="tab:blue", alpha=0.3,
               label="pose guesses")
    for marker_id, pts in log["boxes_before"].items():
        g = np.array([to_global(log["pose_before"], p) for p in pts])
        ax.scatter(g[:, 0], g[:, 1], s=3, color="tab:orange", alpha=0.4,
                   label="box particles (local map)" if marker_id == min(log["boxes_before"]) else None)
    draw_pose(ax, log["pose_before"], "magenta", "estimated pose")
    if log["true_before"] is not None:
        draw_pose(ax, log["true_before"], "red", "true pose (simulation)")

    # 2. the plan, moved into the global frame with the estimated pose
    ax = axes[1]
    draw_world(ax, "2. Planned path")
    for (x, y) in LANDMARKS.values():
        ax.add_patch(Circle((x, y), BOX_RADIUS + ROBOT_RADIUS, color="red", alpha=0.10))
    if log["smooth"] is not None:
        g = np.array([to_global(log["pose_before"], p) for p in log["smooth"]["points"]])
        ax.plot(g[:, 0], g[:, 1], color="black", linewidth=2,
                label=f"smooth path ({log['smooth']['name']}, {log['smooth']['length']:.2f} m)")
    else:
        for i, (a, b) in enumerate(log["rrt_edges"]):
            a, b = to_global(log["pose_before"], a), to_global(log["pose_before"], b)
            ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:green", linewidth=0.7, alpha=0.7,
                    label="RRT tree" if i == 0 else None)
        if log["path"] is not None:
            g = np.array([to_global(log["pose_before"], p) for p in log["path"]])
            ax.plot(g[:, 0], g[:, 1], color="black", linewidth=2, linestyle="--", label="stop-and-turn path (fallback)")
    draw_pose(ax, log["pose_before"], "magenta", "estimated pose")
    if log["goal_pose"] is not None:
        draw_pose(ax, log["goal_pose"], "green", "goal pose")

    # 3. after moving
    ax = axes[2]
    draw_world(ax, "3. After moving: where MIRTE thought it was" if log["drove"] else "3. Not driven (measure only)")
    if log["goal_pose"] is not None:
        draw_pose(ax, log["goal_pose"], "green", "goal pose")
    if log["samples_after"] is not None:
        ax.scatter(log["samples_after"][:, 0], log["samples_after"][:, 1], s=4, color="tab:blue", alpha=0.3,
                   label="pose guesses after driving")
        draw_pose(ax, log["pose_after"], "magenta", "estimate after driving")
    if log["samples_final"] is not None:
        ax.scatter(log["samples_final"][:, 0], log["samples_final"][:, 1], s=4, color="tab:purple", alpha=0.3,
                   label="pose guesses after looking again")
        draw_pose(ax, log["pose_arrival"], "purple", "arrival pose, after looking again")
    if log["true_track"] is not None:
        t = np.array(log["true_track"])
        ax.plot(t[:, 0], t[:, 1], color="red", linewidth=1, label="true path (simulation)")
        draw_pose(ax, log["true_arrival"], "red", "true arrival pose (simulation)")

    # the same view in all three panels
    pts = np.vstack([log["samples_before"][:, :2], np.array(list(LANDMARKS.values())), goal[None]])
    lo, hi = pts.min(axis=0) - 0.8, pts.max(axis=0) + 0.8
    for ax in axes:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("y (m)")
    fig.savefig(filename, dpi=120, bbox_inches="tight")
    print(f"Saved plots to: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def describe(pose):
    return f"x={pose[0]:+.2f} m  y={pose[1]:+.2f} m  heading={math.degrees(pose[2]) % 360:.0f} deg"


def ask_drive(args):
    """True = drive to the midpoint, False = only measure and plot."""
    if args.drive:
        return True
    if args.measure:
        return False
    if not sys.stdin.isatty():
        print("Neither --drive nor --measure was given and there is no keyboard to ask: only measuring.")
        return False
    while True:
        answer = input("Drive to the midpoint, or only measure and plot? With measure MIRTE does not "
                       "move or turn. [d = drive / m = measure]: ").strip().lower()
        if answer in ("d", "drive"):
            return True
        if answer in ("m", "measure"):
            return False


def main():
    ap = argparse.ArgumentParser(description="Exercise 5: localise MIRTE and drive smoothly to the midpoint between two boxes.")
    ap.add_argument("--sim", nargs="*", type=float, metavar="X Y HEADING_DEG",
                    help="run with a simulated MIRTE, optionally at this start pose")
    ap.add_argument("--drive", action="store_true", help="drive to the midpoint without asking")
    ap.add_argument("--measure", action="store_true", help="only measure and plot; MIRTE does not move or turn")
    ap.add_argument("--seed", type=int, help="fixed random numbers, to repeat a run")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "localise_and_drive_smooth.png"),
                    help="where to save the plots")
    args = ap.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    drive = ask_drive(args)

    if args.sim is None:
        robot = MirteRobot()
    else:
        x, y, heading = args.sim if len(args.sim) == 3 else (0.8, -2.2, 60.0)    # in the frame of SIM_BOXES
        robot = SimRobot(x, y, math.radians(heading))
    simulated = isinstance(robot, SimRobot)

    log = dict(samples_after=None, pose_after=None, samples_final=None, pose_final=None, pose_arrival=None,
               smooth=None, goal_pose=None, rrt_edges=[], path=None, drove=False,
               true_track=None, true_arrival=None, true_before=None)
    local_map = {}

    try:
        if drive:
            print("1. Scanning for the boxes")
            found, turned = scan(robot, local_map)
            if not found:
                print(f"Found {len(local_map)} of {NUM_LANDMARKS} boxes within {MAX_LANDMARK_RANGE} m during a full turn. Stopping.")
                return
        else:
            print(f"1. Looking {MEASURE_LOOKS} times without moving")
            readings = measure(robot, local_map)
            if len(local_map) < NUM_LANDMARKS:
                print(f"  {len(local_map)} of {NUM_LANDMARKS} boxes in view. MIRTE needs both in view to work out its pose "
                      "without turning.")
                save_local_plot(local_map, readings, args.out)
                return

        print("2. Aligning the local map with the global map")
        measured = make_global_map(local_map)
        goal_global = np.mean(list(LANDMARKS.values()), axis=0)
        first, second = sorted(LANDMARKS)
        print(f"  landmarks: box {first} at the origin, box {second} at x = {LANDMARKS[second][0]:.2f} m "
              f"(the camera measured {measured:.2f} m between them)")
        pose = align(box_positions(local_map), LANDMARKS)
        log["pose_before"] = pose
        log["samples_before"] = sample_poses(local_map)
        log["boxes_before"] = {i: [(p.getX(), p.getY()) for p in ps] for i, ps in local_map.items()}
        print(f"  MIRTE thinks it is at {describe(pose)}")
        if simulated:
            log["true_before"] = robot.true_pose()
            print(f"  it really is at        {describe(robot.true_pose())}")

        goal_heading = None                     # the arrival heading in the global frame, chosen in round 1
        for round_no in range(1, MAX_ROUNDS + 1):
            print(f"3. Planning (round {round_no})")
            boxes = box_positions(local_map)
            pose = align(boxes, LANDMARKS)
            goal = goal_pose(boxes, None if goal_heading is None else goal_heading - pose[2])
            if goal_heading is None:
                goal_heading = wrap(goal[2] + pose[2])
                log["goal_pose"] = (goal_global[0], goal_global[1], goal_heading)
                print(f"  goal: the midpoint, facing {math.degrees(goal_heading) % 360:.0f} deg")
            smooth = plan_smooth(boxes, goal)
            path = None
            if smooth is not None:
                print(f"  smooth path {smooth['name']}: {smooth['length']:.2f} m, about {smooth['seconds']:.0f} s, "
                      f"{smooth['margin'] * 100:.0f} cm extra clearance")
            else:
                print("  no clear smooth path: falling back to RRT with stop-and-turn")
                edges, rrt_path, path = plan_rrt(boxes, np.array(goal[:2]))
                if path is None:
                    print("  RRT found no path either. MIRTE stays where it is.")
            if round_no == 1:                                       # the plots show the first plan
                log["smooth"] = smooth
                if smooth is None:
                    log["rrt_edges"], log["path"] = edges, path
            if not drive or (smooth is None and path is None):
                break

            print("4. Driving")
            if smooth is not None:
                robot.follow(smooth["commands"])
                move_local_map_along(local_map, smooth["commands"])
            else:
                drive_path(robot, local_map, path)
            log["drove"] = True
            log["pose_after"] = align(box_positions(local_map), LANDMARKS)
            log["samples_after"] = sample_poses(local_map)
            print(f"  MIRTE thinks it is at {describe(log['pose_after'])}, "
                  f"{np.linalg.norm(np.array(log['pose_after'][:2]) - goal_global) * 100:.0f} cm from the midpoint")
            if simulated:
                log["true_arrival"] = robot.true_pose()

            print("5. Looking again")
            found, turned = scan(robot, local_map)
            if not found:
                print("  Did not see every landmark again.")
                break
            log["pose_final"] = align(box_positions(local_map), LANDMARKS)
            log["samples_final"] = sample_poses(local_map)
            # MIRTE turned while looking, so turn the estimated heading back to get the arrival heading
            arrival = (log["pose_final"][0], log["pose_final"][1], wrap(log["pose_final"][2] - turned))
            log["pose_arrival"] = arrival
            miss = np.linalg.norm(np.array(arrival[:2]) - goal_global)
            print(f"  MIRTE thinks it arrived at {describe(arrival)}: {miss * 100:.0f} cm from the midpoint, "
                  f"{math.degrees(wrap(arrival[2] - goal_heading)):+.0f} deg from the goal heading")
            if miss <= GOAL_TOLERANCE:
                break

        if simulated and log["drove"]:
            log["true_track"] = [robot.in_map_frame(x, y)[:2] for x, y in robot.track]
            true = log["true_arrival"]
            print(f"  it really arrived at        {describe(true)}: {robot.true_miss() * 100:.0f} cm from the midpoint, "
                  f"{math.degrees(wrap(true[2] - goal_heading)):+.0f} deg from the goal heading")
    finally:
        robot.stop()

    save_plots(log, args.out)


if __name__ == '__main__':
    main()
