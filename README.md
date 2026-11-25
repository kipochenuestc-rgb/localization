# Map localization

This workspace contains the fixed-map NDT + GNSS + EKF localization package,
its bag preparation tools, camera-pose audits, and RViz configurations.
`D:\slam\slam_lig` remains the SLAM mapping project and provides maps, bag
calibration, sensor recordings, and the matching raw timestamp audit.

Build the NDT node in Ubuntu 24.04 with ROS Jazzy:

```bash
bash scripts/build_ndt.sh
```

Prepare a localization replay from the start of a bag:

```bash
bash scripts/ros_env.sh python3 scripts/prepare_full_localization.py \
  --bag /mnt/d/slam/session_20261124_100923_5cbed2/bag \
  --map-run /mnt/d/slam/slam_lig/runs/new_data_20261124_gnss_buffered \
  --output /mnt/d/slam/localization/runs/ndt_input_start
```

Run a selected time window by adding `--start-offset` and `--duration` to the
preparation command. For example, `--start-offset 120 --duration 60` prepares
the 120–180 second segment. Use a fresh output folder for each window, then run
`complete_full_localization.py` with that input and a distinct run name.

See [docs/ndt_localization.md](docs/ndt_localization.md) for pipeline details,
data assumptions, and result review.
