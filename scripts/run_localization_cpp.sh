#!/usr/bin/env bash
set -eo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SESSION=/mnt/d/slam/session_20261124_100923_5cbed2
BAG=${1:-"$SESSION/bag"}
MAP_RUN=${2:-/mnt/d/slam/slam_lig/runs/new_data_20261124_gnss_buffered}
CALIBRATION=${3:-"$(dirname -- "$BAG")/calibration"}
START_OFFSET=${4:-0}
DURATION=${5:-}
BIAS=${SLAM_GYRO_BIAS_FILE:-"$ROOT/config/profiles/session_20261124_imu_bias.json"}
WORKSPACE=${SLAM_NDT_WORKSPACE:-"$HOME/ndt_ws"}
MAP_FILE="$MAP_RUN/map/SurfMap.pcd"
POSE_FILE="$MAP_RUN/map/transformations.pcd"

if [[ ! -f "$BAG/metadata.yaml" || ! -f "$MAP_FILE" || ! -f "$POSE_FILE" ]]; then
  echo "Bag, map, or SLAM trajectory file is missing." >&2
  exit 1
fi
if [[ ! -f "$WORKSPACE/install/setup.bash" ]]; then
  echo "Build the C++ nodes first with bash scripts/build_ndt.sh." >&2
  exit 1
fi

source /opt/ros/jazzy/setup.bash
source "$WORKSPACE/install/setup.bash"
set -u
export ROS_DOMAIN_ID=${SLAM_ROS_DOMAIN:-${ROS_DOMAIN_ID:-50}}
if [[ $ROS_DOMAIN_ID == 50 ]]; then
  export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT
  export FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT/config/fastdds_local.xml"
else
  export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
fi
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
export LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=4

RUN_DIR=${SLAM_RUN_DIR:-"$ROOT/runs/cpp_$(date -u +%Y%m%d_%H%M%S)_$$"}
mkdir -p "$RUN_DIR"
PIDS=()

# 启动后台节点并把每个进程的输出保存到本次运行目录。
start_node() {
  local name=$1
  shift
  "$@" >"$RUN_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
}

# 退出时结束节点，避免下次回放连接到遗留进程。
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

BIN="$WORKSPACE/install/ndt_localization/lib/ndt_localization"
start_node static_tf "$(ros2 pkg prefix tf2_ros)/lib/tf2_ros/static_transform_publisher" \
  --x 0 --y 0 --z 0 --roll 0 --pitch 0 --yaw 0 \
  --frame-id map --child-frame-id localization_odom
start_node frontend "$BIN/localization_frontend" --ros-args \
  -p "calibration_dir:=$CALIBRATION" -p "map_run:=$MAP_RUN" \
  -p "gyro_bias_file:=$BIAS" -p use_sim_time:=true -p reliable_replay:=true
start_node ekf "$(ros2 pkg prefix robot_localization)/lib/robot_localization/ekf_node" --ros-args \
  --params-file "$ROOT/config/ndt_ekf.yaml" \
  -r odometry/filtered:=/localization/ekf
start_node ndt "$BIN/ndt_localizer" --ros-args \
  -p "map_file:=$MAP_FILE" -p "pose_file:=$POSE_FILE" \
  -p "trajectory_file:=$MAP_RUN/trajectory.csv" -p use_sim_time:=true
if [[ ${SLAM_NO_RVIZ:-0} != 1 ]]; then
  start_node rviz rviz2 -d "$ROOT/config/ndt_localization.rviz"
fi

sleep 3
for pid in "${PIDS[@]}"; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "A localization node exited during startup. Check $RUN_DIR/*.log" >&2
    exit 1
  fi
done

DURATION_ARGS=()
if [[ -n "$DURATION" ]]; then
  DURATION_ARGS=(--playback-duration "$DURATION")
fi
echo "Replaying from $START_OFFSET seconds. Logs: $RUN_DIR"
ros2 bag play -i "$BAG" mcap --disable-keyboard-controls --delay 2 --start-offset "$START_OFFSET" \
  --qos-profile-overrides-path "$ROOT/config/replay_qos.yaml" --wait-for-all-acked 2000 \
  --rate "${SLAM_BAG_RATE:-0.5}" --clock-topics-all \
  "${DURATION_ARGS[@]}" --topics \
  /point_cloud_hasco1 /point_cloud_hasco2 /point_cloud_hasco3 /point_cloud_hasco4 \
  /imu/data /gnss/fix_fixed >"$RUN_DIR/bag.log" 2>&1
if [[ ${SLAM_NO_RVIZ:-0} == 1 ]]; then
  sleep 2
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Localization node exited during playback. Check $RUN_DIR/*.log" >&2
      exit 1
    fi
  done
  exit 0
fi
echo "Bag playback finished. RViz and localization nodes remain active. Press Ctrl+C to stop."
while true; do sleep 60; done
