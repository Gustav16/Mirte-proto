import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
from mcl import MCL
import run_plots
import local_map
import os
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../Mirte/ku_mirte_python"
    )
)

# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True  # Whether or not we are running on MIRTE (camera frames over ROS2), False = laptop camera + keyboard
driveToGoal = True  # On MIRTE: scan, localise and drive to the midpoint between the landmarks. False = stand still and only localise


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    sys.path.append("../../../../Arlo/python")


try:
    from ku_mirte import KU_Mirte

except ImportError:
    print("selflocalize.py: ku_mirte module not present - forcing not running on MIRTE!")
    onRobot = False




# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [4, 2] # IDs of the landmarks in the world
# x = box centre, y = 0 on the line of the front faces (where the markers are, which is what the camera measures to).
landmarks = {
    4: (0.0, 0.0),  # Coordinates for landmark 4
    2: (121.0, 0.0)  # Coordinates for landmark 2: outer edges 150 cm apart, boxes 29 cm wide -> centres 121 cm apart
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks

# MIRTE is always in front of the boxes (y < 0, the side of the markers we measure to). Seeing one box at a time,
# the mirror image of the pose behind the boxes fits the measurements just as well, so the map stops 30 cm
# (about one box depth) behind the face line to rule it out.
m = local_map.LocalMap(low=(-100.0, -250.0),
        high=(500.0, 30.0), landmark_radius=2,
                mirte_radius=2)

m.landmarks = [[list(landmarks[ID]), ID] for ID in landmarkIDs] # same landmarks as above, in the map's format

# The particles are the robot centre (MIRTE turns around it); the camera sits this far in front of it.
CAMERA_OFFSET = 14.0 # cm, from Ex4 (local_map.CAMERA_FORWARD_OFFSET)

# Goal: the midpoint between the two landmarks
GOAL = (np.mean([landmarks[ID][0] for ID in landmarkIDs]), np.mean([landmarks[ID][1] for ID in landmarkIDs]))

# Driving calibration from Ex4 (Execute_path in Ex4/rrt/mirte_rrt.py)
LINEAR_SPEED = 0.4       # m/s. Ex4 used 0.3, but at 0.3 the wheels sometimes did not start: 2 of 3 drives in a logged run didn't move
ANGULAR_SPEED = 0.7      # rad/s
LINEAR_OFFSET = -0.0354 * LINEAR_SPEED / 0.3  # rad/s, keeps MIRTE driving straight (Ex4: -0.0354 at 0.3 m/s, scaled to keep the same curve correction)
TURN_SCALER_LEFT = 1.05
TURN_SCALER_RIGHT = 1.10

# Driving strategy
FRAMES_PER_STOP = 3            # new camera frames used at each stop before the next move
SETTLE_TIME = 0.3              # s, only use frames taken at least this long after a move ended (MIRTE has stopped rocking)
SCAN_STEP = np.radians(25)     # turn left this much between looks while scanning (the camera sees about +-28 deg)
MAX_SCAN_TURNS = 16            # give up after a bit more than a full turn without seeing any landmark
MAX_SPREAD_POS = 15.0          # cm, localised when the median particle is this close to the estimate ...
MAX_SPREAD_THETA = np.radians(10) # ... and its heading this close
MAX_ERROR_DIST = 15.0          # cm, ... and the estimate predicts the measured distances this well
MAX_ERROR_ANGLE = 0.1          # rad, ... and the measured angles this well
MAX_LOOKS = 3                  # stops in a row to stay and look again before turning on
MAX_LEG = 50.0                 # cm, drive at most this far before stopping to look again (only while a landmark is in view)
GOAL_TOLERANCE = 10.0          # cm, close enough to the goal

# Run log: every decision stop is saved to runs/ and plotted by run_plots.py when the program ends
TRUE_START = None              # optional tape-measured start pose (x cm, y cm, theta rad) for the plots, e.g. (60.5, -164.0, np.pi/2)
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")





def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX, 
                                     ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw goal
    goal = (int(GOAL[0] + offsetX), int(ymax - (GOAL[1] + offsetY)))
    cv2.drawMarker(world, goal, CBLACK, cv2.MARKER_CROSS, 12, 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX, 
         ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)



def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(np.random.uniform(m.map_area[0][0], m.map_area[1][0]), np.random.uniform(m.map_area[0][1], m.map_area[1][1]), np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles


def turn(mirte, angle):
    """Turn MIRTE angle (rad, positive = left) on the spot."""
    if abs(angle) < 1e-3:
        return
    scaler = TURN_SCALER_LEFT if angle > 0 else TURN_SCALER_RIGHT
    mirte.drive(0.0, np.sign(angle) * ANGULAR_SPEED, scaler * abs(angle) / ANGULAR_SPEED)


def drive(mirte, dist):
    """Drive MIRTE dist (cm) straight ahead."""
    if dist < 1e-3:
        return
    mirte.drive(LINEAR_SPEED, LINEAR_OFFSET, dist / 100.0 / LINEAR_SPEED)


def pose_spread(particles, est_pose):
    """Median distance (cm) and median heading difference (rad) of the particles to the estimate.
    Medians, so the few random particles that the augmented MCL injects don't count."""
    xs = np.array([p.getX() for p in particles])
    ys = np.array([p.getY() for p in particles])
    thetas = np.array([p.getTheta() for p in particles])
    dist = np.hypot(xs - est_pose.getX(), ys - est_pose.getY())
    dtheta = np.abs(np.mod(thetas - est_pose.getTheta() + np.pi, 2 * np.pi) - np.pi)
    return np.median(dist), np.median(dtheta)


def next_move(est_pose, mcl, seen_ids, z_last, looks_here):
    """Driving strategy. Returns (turn (rad), distance (cm), phase).
    Localised = both landmarks seen, particles close together, and the estimate predicts the landmarks in the last
    frame (z_last) well. The last check catches a particle cloud that has collapsed onto the wrong pose.
    look:  not localised, but a landmark is in view -> stay and look again (at most MAX_LOOKS stops in a row)
    scan:  not localised -> turn left SCAN_STEP and look again
    drive: localised -> turn towards the goal and drive at most MAX_LEG, then look again. With no landmark in view
           (close to the boxes they leave the camera's view) there is nothing to look at, so drive the rest in one go:
           every extra start is a chance for the wheels to stall
    done:  within GOAL_TOLERANCE of the goal"""
    spread_pos, spread_theta = pose_spread(mcl.particles, est_pose)
    z_est = mcl.true_z_for_landmarks(est_pose, m)
    explained = all(abs(z_last[ID][0] - z_est[ID][0]) < MAX_ERROR_DIST and
                    abs(np.mod(z_last[ID][1] - z_est[ID][1] + np.pi, 2 * np.pi) - np.pi) < MAX_ERROR_ANGLE
                    for ID in z_last)
    localised = (set(landmarkIDs) <= seen_ids and spread_pos < MAX_SPREAD_POS and spread_theta < MAX_SPREAD_THETA
                 and explained)
    dx = GOAL[0] - est_pose.getX()
    dy = GOAL[1] - est_pose.getY()
    dist = np.hypot(dx, dy)
    #at the goal the boxes are beside MIRTE, out of view, so the cloud may have spread while driving there blind:
    #allow twice the spread there, but not a cloud that is lost (spread out over the map, mean near the goal by chance)
    if (dist < GOAL_TOLERANCE and set(landmarkIDs) <= seen_ids and explained
            and spread_pos < 2 * MAX_SPREAD_POS and spread_theta < 2 * MAX_SPREAD_THETA):
        return 0.0, 0.0, "done"
    if not localised:
        if z_last and looks_here < MAX_LOOKS:
            return 0.0, 0.0, "look"
        return SCAN_STEP, 0.0, "scan"

    angle = np.arctan2(dy, dx) - est_pose.getTheta()
    angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi # turn the short way
    return angle, (min(dist, MAX_LEG) if z_last else dist), "drive"


# Main program #
try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)


    # Initialize particles
    num_particles = 1000
    particles = initialize_particles(num_particles) #init prior

    aug_mcl = MCL(particles, camera_offset=CAMERA_OFFSET)
    est_pose = aug_mcl.estimate_pose()

    #est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    # Initialize the robot (XXX: You do this)
    if isRunningOnArlo():
        print("Connecting to MIRTE. If this hangs at 'Initiating components', the laptop can't reach MIRTE:"
              " check both are on the same network (ping mirte-f549be.local)")
        mirte = KU_Mirte()
        # Our own camera subscriber (fresh_camera.py): best effort, and with the time each frame was taken,
        # so frames taken before or during a move can be skipped. KU_Mirte's own subscription is stopped,
        # so the images don't come over the network twice.
        from fresh_camera import FreshCamera
        fresh_cam = FreshCamera()
        mirte.executor.add_node(fresh_cam)
        mirte.camera_compressed_sub.destroy_subscription(mirte.camera_compressed_sub.subscription)
        # New ROS2 connections take a while to be found over a hotspot: our camera subscriber, and MIRTE finding
        # KU_Mirte's drive publisher. Drive commands sent before MIRTE listens are silently dropped, which made the
        # first drives of a run stall or come up short. KU_Mirte sends every command on two topics, count both.
        def drive_listeners():
            return (mirte.movement_pub.publisher_mirte.get_subscription_count()
                    + mirte.movement_pub.publisher_gazebo.get_subscription_count())

        print("Waiting for MIRTE's camera frames" + (" and for MIRTE to listen to drive commands" if driveToGoal else ""),
              "(at most 30 s) ...")
        wait_start = last_print = time.time()
        while ((fresh_cam.has_timestamps is None or (driveToGoal and drive_listeners() == 0))
               and time.time() - wait_start < 30.0):
            time.sleep(0.2)
            if time.time() - last_print > 5.0:
                last_print = time.time()
                print("  still waiting, %.0f s: camera frame received: %s, drive listeners: %d"
                      % (time.time() - wait_start, fresh_cam.has_timestamps is not None, drive_listeners()))
        print("Camera timestamps from MIRTE:", fresh_cam.has_timestamps,
              "| MIRTE listens to drive commands:", drive_listeners() > 0, "(after %.1f s)" % (time.time() - wait_start))
        if driveToGoal and drive_listeners() == 0:
            # Seen on MIRTE: with 0 listeners not one of ~20 drive commands moved it. Drive anyway (in case the count
            # is wrong), but say so loudly
            print("\n*** WARNING: nothing on MIRTE listens to drive commands yet - MIRTE will probably not move. ***\n"
                  "*** Check MIRTE's battery and restart MIRTE on the laptop's network if it doesn't move.       ***\n")

    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, aug_mcl.particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        cam = camera.Camera(0, robottype='mirte') # only used for ArUco detection, frames come from mirte
    else:
        #cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=False)

    last_frame = None # last camera frame from MIRTE, to only process new frames

    # Driving state (on MIRTE)
    u_pending = [0.0, 0.0]  # the move [cm, rad] MIRTE made since the last processed frame
    move_end_time = 0.0     # time.time() when the last move ended, older frames are not used
    frames_at_stop = 0      # frames processed since the last move
    seen_ids = set()        # landmarks seen so far
    z_last = {}             # landmarks in the last processed frame
    scan_turns = 0          # scan turns in a row without seeing any landmark
    looks_here = 0          # stops in a row spent looking again without moving
    phase = "scan" if driveToGoal else "standstill"
    start_time = time.time()
    log = {"landmarks": dict(landmarks), "goal": GOAL, "camera_offset": CAMERA_OFFSET,
           "true_start": TRUE_START, "true_end": None, "steps": []}

    while True:

        velocity = angular_velocity = 0.0
        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break
    
        if not isRunningOnArlo():
            if action == ord('w'): # Forward
                velocity += 4.0
            elif action == ord('x'): # Backwards
                velocity -= 4.0
            elif action == ord('s'): # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord('a'): # Left
                angular_velocity += 0.2
            elif action == ord('d'): # Right
                angular_velocity -= 0.2

        u = [velocity, angular_velocity]

        # Fetch next frame
        if isRunningOnArlo():
            colour, captured = fresh_cam.latest()
            if colour is None or colour is last_frame: # no new frame from MIRTE yet, don't reuse the same measurement
                continue
            last_frame = colour
            if captured < move_end_time + SETTLE_TIME: # taken before or during the last move
                continue
            colour = colour.copy() # draw on a copy, not on the cached frame

            # Use motor controls to update particles: the move made since the last processed frame, applied once
            u = u_pending
            u_pending = [0.0, 0.0]
        else:
            colour = cam.get_next_frame()

        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            # List detected objects
            z = {}
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
                # XXX: Do something for each detected object - remember, the same ID may appear several times
                #build observation map, keep the closest detection of each ID (the face pointing at the robot)
                if objectIDs[i] not in z or dists[i] < z[objectIDs[i]][0]:
                    z[objectIDs[i]] = np.array([dists[i], angles[i]])
            z_last = {ID: z[ID] for ID in z if ID in landmarks}
            seen_ids.update(z_last)

            # Compute particle weights
            # XXX: You do this

            # Resampling
            # XXX: You do this
            aug_mcl.mcl(u,z,m)

            # Draw detected objects
            cam.draw_aruco_objects(colour)
        else:
            # No observation - reset weights to uniform distribution
            z_last = {}
             # without a move or a measurement nothing changed (and the motion model adds noise for u = 0)
            aug_mcl.particles = np.array([aug_mcl.sample_motion_model_with_map(u, x_last, m) for x_last in aug_mcl.particles])
            for p in aug_mcl.particles:
                p.setWeight(1.0/num_particles)

    
        est_pose = aug_mcl.estimate_pose() # The estimate of the robots current pose

        if showGUI:

            print(
                "particles:",
                len(aug_mcl.particles),
                "weight range:",
                min(p.getWeight() for p in aug_mcl.particles),
                max(p.getWeight() for p in aug_mcl.particles),
                "pose:",
                est_pose.getX(),
                est_pose.getY(),
                "theta (deg):",
                np.degrees(est_pose.getTheta())
            )
            # Draw map
            draw_world(est_pose, aug_mcl.particles, world)
    
            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)

        # Make the robot drive: after FRAMES_PER_STOP frames at this stop, decide and make the next move
        if isRunningOnArlo() and phase in ("scan", "look", "drive"):
            print("Frames at stop conditon succes")
            frames_at_stop += 1
            if frames_at_stop >= FRAMES_PER_STOP:
                print("frames at stop >= is true")
                frames_at_stop = 0
                angle, dist, phase = next_move(est_pose, aug_mcl, seen_ids, z_last, looks_here)
                looks_here = looks_here + 1 if phase == "look" else 0
                scan_turns = 0 if z_last else scan_turns + (phase == "scan")
                if scan_turns > MAX_SCAN_TURNS:
                    phase = "failed"
                    print("No landmark seen during a full turn - stopping. Seen landmarks:", seen_ids)
                elif phase == "done":
                    print("At the goal", GOAL, "- estimated pose:", est_pose.getX(), est_pose.getY(),
                          "theta (deg):", np.degrees(est_pose.getTheta()))
                else:
                    print(phase, ": turn", np.degrees(angle), "deg, then drive", dist, "cm")
                    turn(mirte, angle)
                    drive(mirte, dist)
                    u_pending = [dist, angle]
                    move_end_time = time.time()

                # Log this stop: copies, since the particles change in place afterwards
                log["steps"].append({
                    "time": time.time() - start_time,
                    "phase": phase,
                    "particles": np.array([[p.getX(), p.getY(), p.getTheta()] for p in aug_mcl.particles]),
                    "estimate": (est_pose.getX(), est_pose.getY(), est_pose.getTheta()),
                    "z": {ID: (float(d), float(a)) for ID, (d, a) in z_last.items()},
                    "move": (angle, dist) if phase in ("scan", "look", "drive") else (0.0, 0.0)})


finally:
    # Make sure to clean up even if an exception occurred

    # Save the run log and its plots (also after q, Ctrl+C or a failure)
    if 'log' in globals() and log["steps"]:
        os.makedirs(RUNS_DIR, exist_ok=True)
        run_path = os.path.join(RUNS_DIR, time.strftime("run_%Y%m%d_%H%M%S"))
        run_plots.save_log(log, run_path + ".pkl")
        run_plots.save_run_plots(log, run_path + ".png")
        print("Saved run log and plots:", run_path + ".pkl/.png")

    # Make sure MIRTE does not keep driving
    if isRunningOnArlo() and 'mirte' in globals():
        try:
            mirte.stop()
        except Exception: # after Ctrl+C, ROS2 is already shut down and can't send the stop; quit with q instead
            print("Could not send stop to MIRTE (ROS2 already shut down by Ctrl+C)")

    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    if 'cam' in globals(): # not created yet if the program was stopped during start-up
        cam.terminateCaptureThread()

