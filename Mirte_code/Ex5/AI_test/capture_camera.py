"""ROS2 still-image capture derived from cameraCalibrations_v4.
Collects images; does NOT compute an intrinsic camera calibration.
ROS imports stay inside main so offline imports need no ROS installation.
"""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np


class FreshFrames:
    def __init__(self):
        self.image=None;self.sequence=0;self.saved_sequence=0;self.received_at=None;self.last_stamp=None
    def receive(self,image,stamp=None):
        if image is None or image.size==0:return False
        if stamp is not None and stamp>0:
            if self.last_stamp is not None and stamp<=self.last_stamp:return False
            self.last_stamp=stamp
        self.image=np.asarray(image).copy();self.sequence+=1;self.received_at=time.monotonic();return True
    def save(self,folder,index,max_age=1.):
        if self.image is None or self.sequence<=self.saved_sequence:
            raise RuntimeError('No new frame since the last save.')
        if time.monotonic()-self.received_at>max_age:raise RuntimeError('Camera frame is stale.')
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'image_{index:04d}.png'
        while path.exists():index+=1;path=folder/f'image_{index:04d}.png'
        if not cv2.imwrite(str(path),self.image.copy()):raise RuntimeError('Could not save camera image.')
        self.saved_sequence=self.sequence
        return path,index+1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('camera_images'))
    p.add_argument('--count',type=int,default=10)
    p.add_argument('--topic',default='/camera/image_raw/compressed')
    args=p.parse_args()
    if args.count<1:p.error('count must be positive')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    store=FreshFrames()
    rclpy.init();node=None
    try:
        node=Node('mirte_still_capture')
        def callback(msg):
            data=np.frombuffer(msg.data,dtype=np.uint8)
            frame=cv2.imdecode(data,cv2.IMREAD_COLOR)
            stamp=msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
            if frame is not None:store.receive(frame,stamp)
        subscription=node.create_subscription(CompressedImage,args.topic,callback,qos_profile_sensor_data)
        del subscription  # Node owns and retains the subscription.
        displayed=None;saved=0;index=0;last_warning=0.;started=time.monotonic()
        print('SPACE: save newest frame; q: quit. Topic:',args.topic)
        while rclpy.ok() and saved<args.count:
            rclpy.spin_once(node,timeout_sec=.02)
            if displayed is None and store.image is not None:displayed=store.image.copy()
            now=time.monotonic()
            if (store.received_at is None and now-started>5 or
                store.received_at is not None and now-store.received_at>1) and now-last_warning>2:
                print('No recent camera frame. Check ROS stream.');last_warning=now
            if displayed is not None:cv2.imshow('MIRTE still image',displayed)
            key=cv2.waitKey(10)&0xFF
            if key==ord('q'):break
            if key==ord(' '):
                try:
                    path,index=store.save(args.output,index);displayed=store.image.copy();saved+=1
                    print(f'{saved}/{args.count}: {path}')
                except RuntimeError as exc:print(exc)
    except KeyboardInterrupt:pass
    finally:
        cv2.destroyAllWindows()
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
