#ifndef NDT_LOCALIZATION__LOCALIZATION_GEOMETRY_HPP_
#define NDT_LOCALIZATION__LOCALIZATION_GEOMETRY_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <deque>
#include <filesystem>
#include <vector>

namespace ndt_localization {

struct Pose3d {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

struct ImuSample {
  double stamp = 0.0;
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Quaterniond integrated_rotation = Eigen::Quaterniond::Identity();
};

struct GeometryConfig {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::array<Eigen::Matrix4d, 4> reference_from_lidar;
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Matrix4d reference_from_camera0 = Eigen::Matrix4d::Identity();
  Eigen::Vector3d antenna_in_camera0 = Eigen::Vector3d::Zero();
  Eigen::Vector3d ecef_origin = Eigen::Vector3d::Zero();
  Eigen::Matrix3d ecef_to_enu = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d enu_to_map = Eigen::Matrix3d::Identity();
  Eigen::Vector3d enu_to_map_translation = Eigen::Vector3d::Zero();

  // 读取当前数据集的雷达外参、陀螺仪偏置和 GNSS 对齐参数。
  static GeometryConfig Load(const std::filesystem::path& calibration_dir,
                             const std::filesystem::path& map_run,
                             const std::filesystem::path& gyro_bias_file);

  // 把 GNSS 经纬高转换到地图坐标系。
  Eigen::Vector3d GnssToMap(double longitude_deg, double latitude_deg,
                            double altitude_m) const;

  // 计算天线原点在定位参考坐标系中的杆臂。
  Eigen::Vector3d AntennaLeverArm() const;
};

class PoseTimeline {
 public:
  // 从 SLAM 位姿点云载入有序的地图轨迹。
  explicit PoseTimeline(const std::filesystem::path& file);

  // 在指定时间插值 SLAM 位姿，只用于定位初始化。
  Pose3d At(double stamp) const;

 private:
  // 读取建图连续里程计的时间、位置和四元数。
  void LoadCsv(const std::filesystem::path& file);

  // 检查时间顺序及位姿有效性，再追加到插值表。
  void Append(double stamp, const Pose3d& pose);

  std::vector<double> stamps_;
  std::vector<Pose3d> poses_;
};

class ImuHistory {
 public:
  // 加入去偏后的陀螺仪测量并递推累计旋转。
  void Add(double stamp, const Eigen::Vector3d& angular_velocity);

  // 判断陀螺仪缓存是否覆盖扫描及其点时间范围。
  bool Covers(double start_stamp, double end_stamp) const;

  // 判断扫描是否早于最早的可用陀螺仪样本。
  bool StartsAfter(double stamp) const;

  // 用相邻陀螺仪样本插值累计旋转。
  Eigen::Quaterniond RotationAt(double stamp) const;

  // 丢弃定位窗口之外的旧样本并保留插值边界。
  void TrimBefore(double stamp);

 private:
  std::deque<ImuSample> samples_;
};

// 读取 ROS 消息头中的秒级时间戳。
double StampSeconds(int32_t sec, uint32_t nanosec);

}  // namespace ndt_localization

#endif  // NDT_LOCALIZATION__LOCALIZATION_GEOMETRY_HPP_
