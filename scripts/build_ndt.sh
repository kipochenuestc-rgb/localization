#!/bin/bash
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/jazzy/setup.bash
WORKSPACE=${SLAM_NDT_WORKSPACE:-$HOME/ndt_ws}
mkdir -p "$WORKSPACE/src"
SOURCE_LINK="$WORKSPACE/src/ndt_localization"
if [ -L "$SOURCE_LINK" ]; then
  CURRENT_SOURCE=$(readlink -f "$SOURCE_LINK" || true)
  if [ "$CURRENT_SOURCE" != "$ROOT/src/ndt_localization" ]; then
    rm "$SOURCE_LINK"
  fi
fi
if [ ! -e "$SOURCE_LINK" ]; then
  ln -s "$ROOT/src/ndt_localization" "$SOURCE_LINK"
elif [ ! -d "$SOURCE_LINK" ]; then
  echo "Unexpected NDT source path: $SOURCE_LINK" >&2
  exit 1
fi
cd "$WORKSPACE"
export CMAKE_BUILD_PARALLEL_LEVEL=2
colcon build --packages-select ndt_localization --executor sequential \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
