MIRTE - RRT + MCL + simple exploration

TWO START PROGRAMS

1) Find a selected ArUco box:
   python3 run_to_box.py

   Enter an ArUco ID, for example:
   10

   Behaviour:
   - take a fresh camera map
   - if target is visible, plan to about 0.40 m in front of it
   - if target is not visible and 2+ boxes are visible, move towards
     a safe passage between two boxes
   - otherwise move a short distance towards a visible box
   - if no safe forward exploration is possible, turn 30 degrees
   - take a new camera map and repeat

   Exploration movements are limited to about 0.40 m at a time.
   The program stops after at most 20 exploration steps.


2) Drive between two selected boxes:
   python3 run_between_boxes.py

   Change these values at the top of run_between_boxes.py:
   BOX_ID_1 = 1
   BOX_ID_2 = 10


SHARED CODE

mirte_rrt_smooth.py  RRT and shared mapping/driving functions
local_map.py         ArUco local map and collision checks
between.py           target, midpoint and passage calculations
path_smoothing.py    simple path shortcutting
path_follower.py     path following
mcl.py               Monte Carlo Localization
robot_models.py      robot model

The remaining Python files are existing support/course files.

IMPORTANT
This is a local exploration strategy, not SLAM. After each exploration
movement MIRTE takes a fresh local map and plans again. This keeps the
implementation simple and close to the original course code.


MIRTE IMPORT
The runnable files add:
../../../Mirte/ku_mirte_python
to sys.path before importing KU_Mirte. This matches the import pattern
used by the working visualize_local_map.py file.

PATH DRAWINGS FOR run_to_box.py
- to_box_plan.png:
  local map and the final planned path to the requested box
- full_search_path.png:
  the complete sequence of planned exploration/path segments from the
  original start until the target is found

The full_search_path image is the route the planner expected to take.
It is not an external ground-truth measurement of the robot trajectory.


V3 CHANGES
- Smooth continuous path following:
  MIRTE now receives non-blocking velocity commands and keeps moving
  while steering/MCL are updated. The old 8 cm stop/start movement is gone.
- Direct path first:
  if the straight route to the local goal is collision-free, RRT is skipped.
- RRT fallback:
  RRT is used only when the direct route is blocked, with max 500 iterations.
- ArUco confirmation:
  local planning maps use two camera frames and keep only IDs visible in both.
  This helps reject one-frame false detections.
