"""ArUco detection/pose compatibility for OpenCV 4 and 5."""
import cv2
import numpy as np

def detect(image, dictionary):
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('OpenCV ArUco is unavailable; install a contrib build.')
    if hasattr(cv2.aruco, 'ArucoDetector'):
        corners, ids, rejected = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary)
    if ids is None or np.asarray(ids).size == 0:
        return corners, None, rejected
    return corners, np.asarray(ids).reshape(-1, 1), rejected

def estimate(corners, marker_length, intrinsic, distortion):
    """Same marker axes as estimatePoseSingleMarkers, using solvePnP."""
    h = marker_length / 2.0
    obj = np.array([[-h,h,0], [h,h,0], [h,-h,0], [-h,-h,0]], np.float64)
    rvecs, tvecs = [], []
    for corner in corners:
        ok, rvec, tvec = cv2.solvePnP(obj, np.asarray(corner,np.float64).reshape(4,2),
                                     intrinsic, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.all(np.isfinite(tvec)) or tvec[2,0] <= 0:
            raise ValueError('Invalid ArUco pose estimate.')
        rvecs.append(rvec.reshape(1,3)); tvecs.append(tvec.reshape(1,3))
    return np.asarray(rvecs), np.asarray(tvecs), obj
