# KU_Mirte Class Documentation

## Overview

`KU_Mirte` is a high-level interface for controlling a Mirte robot in a ROS2 environment. It manages robot actuation, sensor integration, and visualization publishing (e.g., odometry, trees, point clouds, occupancy grids). It wraps multiple ROS2 publishers and subscribers, running in a multi-threaded executor.

---

## Class: `KU_Mirte`

### Constructor

```python
KU_Mirte()
```

Initializes all required ROS2 components:
- Subscribes to: position, camera, lidar, sonar.
- Publishes: odometry, tree, point cloud, occupancy map.
- Starts ROS2 `MultiThreadedExecutor` in a background thread.
- Sets default camera calibration matrices if missing.

---

### Destructor

```python
__del__()
```

Cleans up all ROS2 nodes and shuts down the executor.

---

## Driving

### `set_driving_modifier(speed_modifier: float = 2.4, turn_modifier: float = 3.4)`

Sets modifiers for linear and angular velocity, only affecting real robot driving.

### `drive(lin_speed: float, ang_speed: float, duration: float | None, blocking: bool = True)`

Drives the robot with given linear and angular speed for a specified duration.
- `lin_speed`: meters per second.
- `ang_speed`: radians per second.
- `duration`: seconds (None for indefinite).
- `blocking`: waits for completion if `True`.

### `stop()`

Immediately stops the robot.

### `is_driving -> bool`

Returns whether the robot is currently driving.

---

## Position & Rotation

### `get_position() -> object`

Returns the most recent robot position object with `.x`, `.y`, `.z`.

### `position -> object`

Property alias for `get_position`.

### `get_rotation() -> float`

Returns the robot's rotation (yaw) in radians.

### `rotation -> float`

Property alias for `get_rotation`.

---

## Sensors

### Camera

#### `get_image() -> np.ndarray`

Returns the latest camera image.

#### `image -> np.ndarray`

Property alias for `get_image`.

---

### Lidar

#### `get_lidar_ranges() -> list`

Returns the full 360° lidar scan (list of distances in meters).

#### `lidar -> list`

Property alias for `get_lidar_ranges`.

#### `get_lidar_section(start_angle: float, end_angle: float) -> list`

Returns lidar values between given angles in radians.
- `start_angle`: from -π (left) to π (right)
- `end_angle`: same range

---

### Sonar

#### `get_sonar_ranges() -> dict`

Returns sonar distance readings in meters:
```python
{
  "front_left": float,
  "front_right": float,
  "rear_left": float,
  "rear_right": float
}
```

#### `sonar -> dict`

Property alias for `get_sonar_ranges`.

---

## Visualization

### `set_odometry(reference: str, position: list, rotation: list)`

Sets robot's odometry in either:
- `'mirte'`: robot's local frame
- `'world'`: global map frame

`position`: [x, y, z]  
`rotation`: quaternion [x, y, z, w]

---

### `set_tree(reference: str, edges: list, colours: list | None = None, widths: list | None = None)`

Publishes a tree structure as visualization markers.

- `edges`: list of ((x1, y1), (x2, y2)) pairs
- `colours`: optional list of (r, g, b, a)
- `widths`: optional list of line widths

`reference`: `'mirte'` or `'world'`

---

### `set_occupancy_grid(grid: list, resolution: float, origin: tuple = (0.0, 0.0), rotation: float = 1.0)`

Publishes an occupancy grid for visualization.

- `grid`: 2D list or numpy array with values:
  - 0 = free
  - 100 = occupied
  - -1 = unknown
- `resolution`: size of each grid cell in meters
- `origin`: (x, y) coordinates of grid origin
- `rotation`: orientation in radians

---

### `set_pointcloud(reference: str, points: list, colors: list | None = None)`

Publishes a point cloud.

- `points`: list of (x, y, z)
- `colors`: optional list of (r, g, b, a)
- `reference`: `'mirte'` or `'world'`

---

## Notes

- Some visualizations may require a compatible RViz2 config to appear correctly.

---
