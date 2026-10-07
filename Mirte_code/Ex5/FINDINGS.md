# Ex5 self-localisation: findings (2026-10-07)

Solution: `mcl.py` (particle filter over the robot pose) run by `selflocalize.py`.
For getting the camera feed to the laptop, see [CAMERA_FEED.md](CAMERA_FEED.md).

## Test setup
- Landmarks: box **4** at (0, 0) and box **2** at (121, 0), in cm. x is the box centre, y = 0 is the line of the front faces (the markers).
- Boxes are 29 cm wide; their outer edges are 150 cm apart, so the centres are 121 cm apart.
- MIRTE's camera is 150 cm in front of the faces, centred between the boxes, facing them.
  - Expected camera readings: **161.7 cm, ±0.384 rad (±22°)** to each box.
- The particles are the **robot centre**. The camera is `CAMERA_OFFSET` = 14 cm in front of it.
  - So the expected estimate for this setup is **(60.5, −164, 90°)**.
  - Goal: the midpoint between the landmarks, (60.5, 0).
- Run it on the laptop over ROS2 (`ros-humble` env): `python selflocalize.py`. Press `q` in a window to stop.
  - `driveToGoal = True`: scan, localise and drive to the goal.
  - `driveToGoal = False`: stand still and only localise.

## Results on MIRTE, standing still (iPhone hotspot)
- **Estimate (60.1, −147.0, 90.7°)** against the measured camera position (60.5, −150, 90°), before the camera offset was added. Errors: 0.4 cm, 3 cm and 0.7°. It was close after the first frame and settled within ~15 s.
  - The 0.7° matches the readings: both bearings are shifted by the same ~0.013 rad, so MIRTE really was turned slightly left.
  - The 3 cm comes from distances reading 2–4 cm short (157.7–159.6 vs 161.7 cm). Likely a ~2% scale error in the focal length or marker size, or the tape measurement.
  - Readings barely change from frame to frame (distance ±1 cm, angle ±0.0004 rad). The remaining errors are systematic, not noise.
- With the camera offset, the same readings give a robot-centre estimate of (57.7, −160.7, 89.7°), against (60.5, −164, 90°). This is from the filter offline, not a new robot run.
- Before the fixes (100 cm spacing, 3D angles), it converged to (55, −156, 91.9°).
- When a box shows two faces (e.g. box 4 at 166 cm / 0.277 rad), the closest-detection rule keeps the front face.
- Boxes 6 and 7 far away in the room (8+ m) are detected but ignored, because they are not in the map.

## Driving strategy (`selflocalize.py`, `next_move`)
At each stop, MIRTE uses `FRAMES_PER_STOP` = 3 new frames, then decides:
- **look**: a box is in view but the estimate does not explain it yet. Stay and look again, at most 3 times.
- **scan**: turn left 25° and look again. Stop after a full turn without seeing any box.
- **drive**: localised. Turn towards the goal and drive at most 50 cm, then look again.
- **done**: within 10 cm of the goal.

"Localised" means all three of:
- both boxes have been seen;
- the median particle is within 15 cm and 10° of the estimate;
- the estimate predicts the boxes in the last frame within 15 cm and 0.1 rad.

The first frame after each move is skipped, since it may have been taken while MIRTE was still moving. Each move is given to the filter once, as `u = [cm, rad]`. Turning and driving use the Ex4 calibration (0.3 m/s, 0.7 rad/s, scalers 1.05 / 1.10, offset −0.0354).

**On MIRTE (test setup, centred, 1.5 m):** ended cleanly in the middle between the boxes. The world view and the camera feed agreed along the way.
- It looked twice before driving: the first estimate, (97, −167, 108°), was rejected by the consistency check.
- It drove from (85, −162, 99°). That was still off along the circle around the boxes, but the direction to the goal was right, so it only turned −0.7°.
- After the first 50 cm leg, the estimate snapped to (57, −127, 90°).
- Below ~80 cm from the boxes they leave the camera's view. The last ~85 cm were driven on the motion model alone, so the Ex4 drive calibration holds.
- Final estimate: (60.6, 0.2, 87°).

**Second drive on MIRTE (`runs/run_20261007_131604`):** ended with the camera on the face line, so the robot centre was ~14 cm short of the goal. The estimate said (61, −0.2).
- The log shows that **2 of the 3 checkable 50 cm drives didn't move MIRTE at all**: identical camera readings before and after. The third moved ~40 cm.
- The filter caught this each time (the camera pulled the estimate back).
- The last 136 cm were blind (no box in view), so the shortfall there stayed in the end position.
- Changes: `LINEAR_SPEED` 0.3 → 0.4 m/s, with `LINEAR_OFFSET` scaled to match. Once no box is in view, the rest is driven in one go instead of 50 cm legs.

**Third drive on MIRTE (`runs/run_20261007_133804`, 0.4 m/s):** reached the goal, estimate (60.1, 0.2), but arrived at a 75° heading.
- **The first 50 cm drive stalled again** (identical readings before and after). The later drives worked. Across both logged runs, the stalls hit the first drive(s) of a run.
- Next: check every drive with wheel odometry (`/odom`), retry if the wheels didn't move, and feed the measured motion to the filter.

**Simulated** (real `next_move` / MCL / map, simulated MIRTE with 5% turn and 3% drive errors, the camera's FOV, and the 2% short distances):
- 60/60 runs reached the goal, from 5 start poses including facing away from the boxes.
- Median true distance to the goal: 4–9 cm; worst run 20 cm.

## Run logs and plots
- Every decision stop is logged: phase, particles, estimate, measurements, commanded move.
- When `selflocalize.py` ends (`q`, Ctrl+C, done or failed), it saves `runs/run_<date>_<time>.pkl` and `.png`.
- Set `TRUE_START` to the tape-measured start pose to see it in the plots.
- Re-plot a saved run: `python run_plots.py runs/<run>.pkl`. The plotting code is in `run_plots.py`.

## Bugs found and fixed
| Where | Problem | Fix |
|---|---|---|
| `mcl.py` `p_hit` | Only distance was used, so θ could not converge | Multiply in a Gaussian on the wrapped bearing error (eq. 1, 3) |
| `mcl.py` `sigma_hit` | σ_bearing = 0.015 rad (0.9°) was far too tight | 0.1 rad (~6°), to be tuned |
| `mcl.py` `mcl()` | Standing still adds no motion noise, so after resampling all particles collapse into one | Jitter after resampling: `add_uncertainty`, σ = 2 cm / 0.02 rad |
| `mcl.py` `sample_motion_model_with_map` | With u = 0, a particle in collision (outside the map, or within 4 cm of a landmark) was redrawn forever. The script hung in ~1 of 4 runs before any window opened | After 10 tries, replace the particle with a random free pose |
| `mcl.py` `sample_motion_model` | The turn was applied twice (`θ + u[1] + d_rot_1_est`) | Turn by `d_rot_1_est`, then drive along the new heading |
| `mcl.py` alphas | `u` is in cm, so α₂ = 0.05 gave ~11 rad of heading noise after a 50 cm drive | Values with units: α₁ = 0.01, α₂ = 3e-7 rad²/cm², α₃ = 0.0025, α₄ = 0.5 cm²/rad² (to be tuned) |
| `mcl.py` `true_z_for_landmarks` | Turning on the spot rotates around the robot centre, not the camera | Predict measurements from the camera, `camera_offset` in front of the particle |
| `mcl.py` augmented MCL | With 0.1 / 0.001, w_slow stayed at the poor first frames, so no particles were ever injected | `fast_const` 0.3, `slow_const` 0.02, and compare the average weight per seen landmark |
| `mcl.py` injection | Uniform random particles almost never land near the right pose, so seeing one box at a time made the cloud collapse onto the wrong spot | Sensor resetting: `sample_from_measurement` injects poses that fit a seen box |
| `camera.py` | Angle and distance were 3D, so a marker ~22 cm above the camera read ~1.5° too wide | Use the horizontal part only: `-atan2(x, z)`, `hypot(x, z)` |
| `camera.py` | Crashed with NumPy 2.5 (array element assignment) | Replaced by the code above |
| `camera.py` | No MIRTE calibration | `robottype='mirte'`: f = 609.9 px, 640×480, 0.145 m marker, no local camera opened |
| `selflocalize.py` + new `fresh_camera.py` | Frames taken before or during a move arrived late over the hotspot (reliable delivery, no timestamps) and were used after the move. Boxes were missed, and the particles spread into a ring around one box | Own camera subscriber: best effort, with capture times; only frames taken ≥ 0.3 s after the move are used (see CAMERA_FEED.md) |
| `selflocalize.py` | `draw_aruco_objects` drew on KU_Mirte's cached frame, so the next loop detected nothing and the pose froze | Only process *new* frames, and draw on a copy |
| `selflocalize.py` | A box seen from two sides used the first detection, which could be the side face | Keep the closest detection of each ID |
| `selflocalize.py` | Landmarks 100 cm apart | 121 cm (measured) |

## Still open
- **First MIRTE run with `fresh_camera.py`.** Tested only against a fake camera on the laptop. Check that it prints `Camera timestamps from MIRTE: True`.
- **Check the 0.4 m/s drive on MIRTE.** Does it start every time, and does a straight leg stay straight (scaled `LINEAR_OFFSET`)? Do turns of a few degrees move at all?
- **Test other start poses on MIRTE**, e.g. off to one side or facing away, to exercise the scan.
- **Stop with `q`, not Ctrl+C.** Ctrl+C shuts down KU_Mirte's ROS2 first, so the final `mirte.stop()` can't be sent.
- **Paths are straight lines.** They do not avoid the boxes, which is fine when starting in front of them.
- **The goal is on the face line** (y = 0). To end between the box centres instead, use y = 14.5.
- **Tune σ_d / σ_θ and the motion alphas** from real runs.
- **Hotspot frame rate**, see [CAMERA_FEED.md](CAMERA_FEED.md).
