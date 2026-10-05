# Exercise 5 — Self Localization

This folder separates **Exercise 5 localization** from the Exercise 4 local-RRT code.

## What the official exercise requires

Two known ArUco landmarks have a known distance between them. MIRTE must:

1. estimate its pose `(x, z, theta)` in the landmark coordinate frame,
2. use distance and orientation/bearing observations to the landmarks,
3. use particle filtering / Monte Carlo Localization,
4. drive to the midpoint between the two landmarks.

## Final file roles

- `ex5_config.py` — IDs, known landmark distance, calibration and motion/noise parameters.
- `aruco_measurements.py` — one camera frame -> `(id, range, bearing)` measurements from the robot center.
- `mcl.py` — the authoritative particle-filter implementation for Exercise 5.
- `run_between_boxes.py` — **main program**: initial localization, closed-loop movement, repeated MCL updates, stop at midpoint.
- `simulate_mcl.py` — offline mathematical sanity test; no robot needed.
- `camera.py`, `framebuffer.py` — original course camera support; retained for reference/support, but the main solution uses `KU_Mirte.get_image_compressed()` because that path already worked in Exercise 3/4.
- `particle.py`, `random_numbers.py` — original course particle helpers. They are retained, but `mcl.py` uses a NumPy particle array instead. Do not run two different particle-filter implementations at the same time.

## Files from the AI-test folder that are *not* part of the core Exercise 5 solution

The following are mainly Exercise 4 / extra exploration functionality:

- `local_map.py`
- `between.py`
- `mirte_rrt_smooth.py`
- `path_follower.py`
- `path_smoothing.py`
- `robot_models.py`
- `grid_occ.py`
- `run_to_box.py`
- `visualize_local_map.py`

They can remain in your project, but they should not define the coordinate frame for Exercise 5 MCL.

## Before the physical test

Edit `ex5_config.py`:

1. `LANDMARK_ID_A` and `LANDMARK_ID_B` must match the two boxes.
2. Measure the distance **between the two landmark reference points** and set `LANDMARK_DISTANCE_M`.
3. Verify `CAMERA_FORWARD_OFFSET_M`.
4. Verify the ArUco marker side length (`145 mm` in the current setup).

The world frame is defined as:

- landmark A: `(0, 0)`
- landmark B: `(D, 0)`
- midpoint goal: `(D/2, 0)`
- `theta = 0`: +z direction
- positive theta/bearing: left

## Offline test

```bash
python3 simulate_mcl.py
```

It should converge close to the simulated true pose.

## Robot test

```bash
python3 run_between_boxes.py
```

Start with both selected ArUco markers visible. The program takes several stationary observations first, then alternates between a small rotate/translate action and a new MCL observation update.

## Why this differs from the old AI-test implementation

The old implementation created a new local map with MIRTE at `(0,0,0)` and then created a fresh MCL for each local path. That is useful for local path tracking, but it is not the same as estimating MIRTE's pose in a fixed landmark coordinate frame. Exercise 5 requires the latter.


## Exercise 4 geometry values now reused here

The camera/ArUco geometry has been aligned with the latest tested Exercise 4
implementation:

- ArUco marker size: **145 mm**
- focal length used by the 640x480 calibration: **609.9 px**
- camera forward offset from MIRTE centre: **+0.14 m**
- physical box depth: **0.25 m**
- MIRTE collision radius used in Ex4: **0.22 m**
- landmark/box half-width approximation used in Ex4: **0.20 m**

The important sign discovered in Exercise 4 is preserved exactly:

```python
box_center_camera = tvec + R @ np.array([
    0.0,
    0.0,
    -BOX_DEPTH_MM / 2.0,
])
```

The minus sign is deliberate. The older AI-test rectangle code instead chose
marker `+Z`/positive map-z and could therefore place the physical box on the
wrong side of the ArUco plane.

### Important distinction for Exercise 5

MCL still measures **the ArUco marker centre**, because the fixed landmark map
and the visual observation must describe the same physical point. Therefore the
`-BOX_DEPTH/2` correction is **not** applied to the range/bearing measurement
used by MCL. It is only applied when we need the physical box centre for
plotting, collision geometry, or debugging.

Before driving, run:

```bash
python3 geometry_check.py
```

This sends no motor commands. It prints both the ArUco position and inferred
box centre so the sign can be checked on the real setup.
