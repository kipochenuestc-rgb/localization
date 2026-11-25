# 固定地图定位

定位前端和 NDT 节点使用 C++17。前端同步四路雷达，使用去偏后的 IMU 角速度补偿扫描内旋转。NDT 载入完整的 `SurfMap.pcd`，按当前位置从地图裁取半径 90 m 的局部点云进行配准。C++ `robot_localization` EKF 融合通过质量检查的 NDT 六自由度位姿、IMU 三轴角速度和杆臂补偿后的 GNSS 水平位置。

## 运行回放

在 Ubuntu 24.04 和 ROS 2 Jazzy 中执行以下命令，从原始 bag 第 280 秒开始播放 30 秒。最后两个参数分别为起点和时长，单位为秒。默认启用 RViz，显示全图、配准点云、融合轨迹和 GNSS 轨迹。

```bash
cd /mnt/d/slam/localization
bash scripts/build_ndt.sh
session=/mnt/d/slam/session_20261124_100923_5cbed2
map_run=/mnt/d/slam/slam_lig/runs/new_data_20261124_gnss_buffered
bash scripts/run_localization_cpp.sh "$session/bag" "$map_run" "$session/calibration" 280 30
```

每次回放按首帧扫描的时间戳在优化关键帧间插值，超出关键帧时间范围时改从同次建图的 `trajectory.csv` 插值。建图轨迹只用于提供初值，之后使用传感器和滤波结果更新位姿。相同录制数据上的回放可评估一致性，绝对精度需要独立参考。

设置 `SLAM_NO_RVIZ=1` 可在回放结束后自动退出，`SLAM_ROS_DOMAIN` 可隔离同时运行的回放。运行日志位于 `runs/cpp_*`，`ndt.log` 包含每帧的接受状态、内点比例和拟合误差。

已有准备数据的相机导出及离线审核流程见 [定位和相机工具说明](docs/ndt_localization.md)。
