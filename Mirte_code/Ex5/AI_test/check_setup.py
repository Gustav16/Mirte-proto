"""Environment check; --robot reads sensors but NEVER commands movement."""
import argparse
import hashlib
import inspect
import time
import cv2
import numpy as np
import ex5_config as cfg
from robot_io import KU_Mirte,require_continuous_api


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--robot',action='store_true')
    args=parser.parse_args()
    print('OpenCV:',cv2.__version__,'ArUco:',hasattr(cv2,'aruco'))
    print('Landmarks:',cfg.LANDMARKS,'Goal:',cfg.GOAL)
    print('Driver modifiers:',cfg.DRIVE_SPEED_MODIFIER,cfg.DRIVE_TURN_MODIFIER)
    print('Physical response gains:',cfg.DRIVE_LINEAR_GAIN,cfg.DRIVE_LEFT_GAIN,cfg.DRIVE_RIGHT_GAIN)
    print('Low-speed operation verified:',cfg.DRIVE_LOW_SPEED_VERIFIED)
    if not cfg.DRIVE_LOW_SPEED_VERIFIED:
        print('Old code reports a low-speed dead zone. Verify smooth settings with calibrate_drive.py before navigation.')
    print('Configured camera:',cfg.IMAGE_WIDTH,cfg.IMAGE_HEIGHT,'f=',cfg.FOCAL_LENGTH_PX)
    if not args.robot:
        print('Offline only. Run python3 test_project.py for software checks.');return
    robot=KU_Mirte()
    print('drive signature:',inspect.signature(robot.drive))
    require_continuous_api(robot)
    print('Continuous-command arguments accepted; internal driver behavior still requires verification.')
    print('sonar readings (verify units against measured physical distances):',robot.sonar)
    seen=set()
    for _ in range(10):
        image=robot.get_image_compressed()
        if image is None: print('No image.')
        else:
            print('Image shape:',image.shape)
            if image.shape[:2]!=(cfg.IMAGE_HEIGHT,cfg.IMAGE_WIDTH):raise RuntimeError('Image calibration mismatch.')
            seen.add(hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest())
        time.sleep(.3)
    print('Different image contents in 10 captures:',len(seen))
    if len(seen)<2:print('Image stream may be frozen. Verify updating images before driving.')


if __name__=='__main__':main()
