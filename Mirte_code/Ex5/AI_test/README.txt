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
