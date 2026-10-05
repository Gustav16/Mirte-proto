"""Thread-safe latest-frame store; callers receive an owned copy."""
import threading
import numpy as np

class FrameBuffer:
    def __init__(self,frame_shape):
        self.frameShape=tuple(frame_shape);self.lock=threading.Lock();self._frame=None
        self.sequence=0
    def get_frame(self):
        with self.lock:return None if self._frame is None else self._frame.copy()
    def new_frame(self,frame):
        frame=np.asarray(frame)
        if frame.shape!=self.frameShape or frame.dtype!=np.uint8:
            raise ValueError('Unexpected camera frame shape/dtype.')
        with self.lock:self._frame=frame.copy();self.sequence+=1
