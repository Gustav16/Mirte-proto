from pathlib import Path
import sys

EX6_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EX6_DIR))

import camera
from ku_mirte import KU_Mirte

landmarks = {
    1: (0.0, 0.0),
    2: (0.0, 300.0),
    3: (400.0, 0.0),
    6: (300.0, 400.0),
}

mirte = KU_Mirte()
z, local_landmarks, local_obstacles = camera.get_observations(landmarks, mirte)

target_id = 6  # Skift til ID'et på markøren, I tester

print("Landmarks fundet:", local_landmarks.keys())
print("Observationer:", z)

if target_id in local_landmarks:
    print(f"Target {target_id} er fundet")
else:
    print(f"Target {target_id} er ikke fundet")