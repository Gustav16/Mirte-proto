# MIRTE camera live feed on the laptop (ROS2)

`selflocalize.py` runs on the laptop. It gets MIRTE's camera images over ROS2, from the topic `/camera/image_raw/compressed`, through its own subscriber in `fresh_camera.py`. `KU_Mirte` is still used for driving.

## Getting it running
1. **Same network.** MIRTE and the laptop must be on the same network. If the usual network is not available, use a phone hotspot: connect MIRTE to it, then join it with the laptop. This worked on an iPhone hotspot (laptop at 172.20.10.x).
2. **Check MIRTE is reachable:** `ping mirte-f549be.local`. On the hotspot, expect 150–500 ms.
3. **Python environment:** `conda activate ros-humble`. It has `rclpy`, `cv_bridge` and an OpenCV that can show windows. The `rex` env has no `rclpy`.
4. **Run:**
   ```bash
   cd Mirte_code/Ex5
   python selflocalize.py
   ```
   Two windows open: **Robot view** (the camera, with detected markers drawn) and **World view** (particles, landmarks, goal ✚, estimate). Press `q` in a window to stop.

## What in the code makes it work
- **`camera.Camera(0, robottype='mirte')`** only does ArUco detection with MIRTE's calibration (f = 609.9 px, 640×480, 0.145 m marker). It does *not* open the laptop's camera.
- **`fresh_camera.py` (`FreshCamera`) replaces `mirte.get_image_compressed()`.** KU_Mirte's subscriber uses reliable delivery and throws away the frame's timestamp. On the hotspot, a lost packet was resent, so a frame taken *before or during* a move could arrive seconds later and be used as if it were new. The particles were moved by the turn but weighed against the old view, which missed boxes and spread the particles into a ring. `FreshCamera`:
  - uses **best-effort** delivery: a damaged frame is dropped instead of delaying the newer ones;
  - keeps each frame's **capture time**, converted to the laptop's clock (works even if the robot's clock is off).
  - `selflocalize.py` only uses frames taken at least `SETTLE_TIME` = 0.3 s after the last move ended.
  - KU_Mirte's own camera subscription is stopped, so images don't come over the network twice.
  - At start-up it prints `Camera timestamps from MIRTE: True/False`. If False, it falls back to arrival time, and old frames are possible again.
- **Only new frames are processed.** A frame that is the same object as last time is skipped. Without this, one frame gets fed to the filter many times.
- **Drawing happens on a copy** (`colour.copy()`). Drawing marker outlines on the cached image made the next detection fail, and the feed looked frozen.
- **NumPy 2.5:** `camera.py` no longer assigns a 1×1 array to an array element. That used to crash in `ros-humble`.

## Hangs at "Initiating components"
`KU_Mirte()` waits, with no timeout, until MIRTE's camera has sent its calibration and a first image. If it hangs there, the laptop can't reach MIRTE. Almost always this means the laptop and MIRTE are on different networks, e.g. after switching Wi-Fi.
- Stop with Ctrl+C.
- Check with `ping mirte-f549be.local`. If it doesn't answer, put the laptop on the same network as MIRTE.
- If ping works but it still hangs, ROS2 on MIRTE is not visible from the laptop. `ros2 topic info /camera/image_raw/compressed` then shows `Publisher count: 0`. MIRTE's ROS2 software picks its network when it starts. If MIRTE changed Wi-Fi afterwards, go back to the network it started on (this happened with Gustav's hotspot vs the iPhone hotspot), or restart MIRTE on the new network.

## When the feed is slow or stops
- **Check the actual frame rate** in a second terminal (`ros-humble` env): `ros2 topic hz /camera/image_raw/compressed`
- **Hotspot gaps of 0.1–7 s between frames are normal.** It got noticeably faster once the main loop stopped re-processing old frames, which leaves time for KU_Mirte's receive thread.
- **If it is still too slow:** lower the JPEG quality/fps on the robot. Best-effort delivery is already used.
- **Robot view freezes during a move:** expected. The windows only update between moves.

## Messages you can ignore
- `[WARN] ... Publisher already registered for provided node name`: from KU_Mirte when it starts.
- `drawFrameAxes Some of projected axes endpoints are out of frame`: a marker near the image edge.
- A `KU_Mirte.__del__ ... rclpy ... shutdown` traceback after `q` or Ctrl+C: cleanup while Python is exiting.
