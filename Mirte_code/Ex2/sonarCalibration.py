import sys
import os
import math
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Mirte/ku_mirte_python'))
from Mirte.ku_mirte_python.ku_mirte import KU_Mirte


mirte = KU_Mirte()


SCANNING_TIME = 5
SAMPLE_INTERVAL = 0.05  # seconds between sonar samples

front_left = []
front_right = []
rear_left = []
rear_right = []


def save_readings():
    """Write each sonar array to its own txt file next to this script."""
    readings = {
        'front_left': front_left,
        'front_right': front_right,
        'rear_left': rear_left,
        'rear_right': rear_right,
    }
    out_dir = os.path.join(os.path.dirname(__file__), 'sonarData')
    os.makedirs(out_dir, exist_ok=True)
    for name, values in readings.items():
        path = os.path.join(out_dir, f'sonar_{name}.txt')
        with open(path, 'w') as f:
            f.write('\n'.join(str(v) for v in values))
        print(f'saved {len(values)} readings to {path}')


try:
    while True:
        msg = input('press (q) to quit, or press any other key to scan for 5 seconds:\n')
        if msg.lower() == 'q':
            break

        scan_end = time.time() + SCANNING_TIME
        while time.time() < scan_end:
            front_left.append(mirte.sonar['front_left'])
            front_right.append(mirte.sonar['front_right'])
            rear_left.append(mirte.sonar['rear_left'])
            rear_right.append(mirte.sonar['rear_right'])
            time.sleep(SAMPLE_INTERVAL)

    save_readings()

finally:

    mirte.stop()
    del mirte
