# Fixed-map NDT + GNSS + EKF localization

This pipeline uses the four LiDAR clouds, bias-corrected IMU angular velocity,
GNSS and a saved surface map. Only the first saved SLAM pose initializes the
localizer. Later saved SLAM poses are used in the audit, never as localization
measurements. This is initialized map localization, not global place recognition.

All transforms use `map_from_reference`; the reference origin is LiDAR 1 with
IMU axes. Camera optical axes are x right, y down, z forward. Camera transforms
come from the same session calibration used by the RGB pipeline.

The default raw-bag replay uses the C++ front end, NDT and EKF. The [project
README](../README.md) gives the build command and an example with a start offset.

## Raw-bag replay results

Five independent 30-second clips from `session_20261124_100923_5cbed2` were
replayed at half speed with the full `new_data_20261124_gnss_buffered` surface
map. Each clip contains the four original LiDAR topics, raw IMU and fixed GNSS.
The denominator is the number of four-LiDAR groups with valid timestamp and IMU
coverage in the extracted data. GNSS counts measure publications to the EKF input.

| Bag interval (s) | Accepted NDT / available | GNSS to EKF | EKF poses | Horizontal difference P95 (m) |
| --- | ---: | ---: | ---: | ---: |
| 0–30 | 194 / 194 | 25 | 1441 | 0.313 |
| 150–180 | 193 / 193 | 5 | 1410 | 0.327 |
| 280–310 | 193 / 193 | 29 | 1443 | 0.379 |
| 470–500 | 193 / 193 | 26 | 1407 | 0.283 |
| 730–760 | 194 / 194 | 28 | 1429 | 0.022 |

The horizontal difference compares EKF outputs to linearly interpolated XY
positions from the same mapping run. The first four clips use optimized
keyframes. The last clip is beyond the final keyframe at about 689.5 s and uses
the continuous mapping trajectory for its initial seed and the offline comparison.
These figures measure replay consistency. Absolute positioning accuracy and
real-time throughput require separate tests.

## Offline camera export in Ubuntu-24.04 / ROS Jazzy

Run from `/mnt/d/slam/localization` after building the `src/ndt_localization`
package in `/home/kipoc/ndt_ws`. The adjacent `slam_lig` checkout supplies the
SLAM map, calibration, and bag.
Use a fresh output folder for each run. Never run two localization controllers
in the same ROS domain simultaneously.

```bash
bash scripts/build_ndt.sh

bash scripts/ros_env.sh python3 scripts/prepare_full_localization.py \
  --bag /mnt/d/slam/session_20261124_100923_5cbed2/bag \
  --map-run /mnt/d/slam/slam_lig/runs/new_data_20261124_gnss_buffered \
  --output /mnt/d/slam/localization/runs/ndt_input_full

bash scripts/ros_env.sh python3 scripts/complete_full_localization.py \
  --input /mnt/d/slam/localization/runs/ndt_input_full --run ndt_ekf_full
```

The preparation script reads IMU/GNSS/image timestamps, then streams complete
four-LiDAR groups with bounded memory. Use `--start-offset` and `--duration` to
select any recording interval. Each point is rotated to the LiDAR 1
header time using bias-corrected gyro integration. Translation deskew is not yet
implemented. The scan is voxel filtered before NDT. Point time units and the
assumption that the header denotes scan start are inherited from the current
sensor adapter.

NDT uses a cached local submap, a 1 m distribution grid and 0.65 m source voxels.
It stops on small squared translation AND small angular change, or an iteration
limit. Reaching the limit is rejected. Current acceptance gates are fitness less
than 0.3 m², nearest-map inliers above 75%, initial-to-final translation below 2 m
and rotation below 0.2 rad. These are operational gates, not accuracy guarantees.
Prediction uses accepted NDT displacement and EKF orientation; velocity is not
an external wheel-odometry measurement.

The EKF fuses NDT pose, IMU gyro and optional GNSS XY. GNSS uses the saved map
alignment, applies the antenna lever arm and rejects a horizontal innovation
above 3 m. The existing map's GNSS factors did not apply that lever arm, so this
does not retroactively correct its map geometry or georeferencing.

## Results and review

`ndt_quality.jsonl` records acceptance, iterations, raw NDT pose, map fitness,
inliers and computation time. `ekf_trajectory.csv` contains all filter outputs.
The camera exporter interpolates those outputs at the original image message
timestamps and composes each camera's extrinsics. It never extrapolates outside
the filter's time support. Missing poses, no accepted NDT within 0.3 s and a
position covariance trace above 0.5 m² are explicit review flags.

The exported `all_camera_poses.csv` is in the map frame. `all_camera_poses.npz`
preserves all original image groups, including NaNs for unsupported poses.
The four images in a mosaic have one shared message timestamp; individual
exposure times were not supplied.

RViz shows the saved map, registered scan, estimated path, camera frusta and
original camera images. Live frusta refer to the current scan time; the image
offset (normally about 2 ms) is displayed. Saved camera playback uses the actual
image timestamp. The four-camera display is camera 0/1 on top, 2/3 below.

These are same-recording map replays. The map and GNSS are not independent
ground truth. GNSS residuals and agreement with saved SLAM poses measure
consistency, not surveyed absolute accuracy. Camera extrinsics remain a candidate
calibration. Good visual alignment is useful evidence, not subpixel calibration.
