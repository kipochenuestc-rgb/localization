#include "localization_geometry.hpp"

#include <pcl/PCLPointCloud2.h>
#include <pcl/PCLPointField.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ndt_localization {
namespace {

// 把 JSON 或 YAML 中的三元素数组读成 Eigen 向量。
Eigen::Vector3d ReadVector3(const YAML::Node& node) {
  if (!node.IsSequence() || node.size() != 3) {
    throw std::runtime_error("期望包含三个数值的数组");
  }
  return {node[0].as<double>(), node[1].as<double>(), node[2].as<double>()};
}

// 把三行三列的旋转矩阵转换为 Eigen 矩阵。
Eigen::Matrix3d ReadMatrix3(const YAML::Node& node) {
  if (!node.IsSequence() || node.size() != 3) {
    throw std::runtime_error("旋转矩阵必须是三行矩阵");
  }
  Eigen::Matrix3d matrix;
  for (int row = 0; row < 3; ++row) {
    if (!node[row].IsSequence() || node[row].size() != 3) {
      throw std::runtime_error("旋转矩阵每行必须包含三个数值");
    }
    for (int col = 0; col < 3; ++col) {
      matrix(row, col) = node[row][col].as<double>();
    }
  }
  return matrix;
}

// 把四行四列的外参矩阵转换为 Eigen 矩阵。
Eigen::Matrix4d ReadMatrix4(const YAML::Node& node) {
  if (!node.IsSequence() || node.size() != 4) {
    throw std::runtime_error("雷达外参必须是四行矩阵");
  }
  Eigen::Matrix4d matrix;
  for (int row = 0; row < 4; ++row) {
    if (!node[row].IsSequence() || node[row].size() != 4) {
      throw std::runtime_error("雷达外参每行必须包含四个数值");
    }
    for (int col = 0; col < 4; ++col) {
      matrix(row, col) = node[row][col].as<double>();
    }
  }
  return matrix;
}

// 按 PCD 字段类型读取一个数值，供位姿时间轴解析使用。
double ReadPcdValue(const pcl::PCLPointCloud2& cloud, uint32_t point,
                    const pcl::PCLPointField& field) {
  const uint8_t* data =
      cloud.data.data() + point * cloud.point_step + field.offset;
  switch (field.datatype) {
    case pcl::PCLPointField::FLOAT32: {
      float value;
      std::memcpy(&value, data, sizeof(value));
      return value;
    }
    case pcl::PCLPointField::FLOAT64: {
      double value;
      std::memcpy(&value, data, sizeof(value));
      return value;
    }
    default:
      throw std::runtime_error("位姿 PCD 字段不是浮点数");
  }
}

// 查找位姿 PCD 中指定名称的字段。
const pcl::PCLPointField& FindField(const pcl::PCLPointCloud2& cloud,
                                    const std::string& name) {
  const auto found = std::find_if(
      cloud.fields.begin(), cloud.fields.end(),
      [&name](const pcl::PCLPointField& field) { return field.name == name; });
  if (found == cloud.fields.end()) {
    throw std::runtime_error("位姿 PCD 缺少字段：" + name);
  }
  return *found;
}

// 按 WGS84 椭球将经纬高转换为地心地固坐标。
Eigen::Vector3d GeodeticToEcef(double longitude_deg, double latitude_deg,
                               double altitude_m) {
  constexpr double kSemiMajorAxis = 6378137.0;
  constexpr double kEccentricitySquared = 6.6943799901413165e-3;
  constexpr double kPi = 3.14159265358979323846;
  const double longitude = longitude_deg * kPi / 180.0;
  const double latitude = latitude_deg * kPi / 180.0;
  const double sin_latitude = std::sin(latitude);
  const double radius =
      kSemiMajorAxis /
      std::sqrt(1.0 - kEccentricitySquared * sin_latitude * sin_latitude);
  return {(radius + altitude_m) * std::cos(latitude) * std::cos(longitude),
          (radius + altitude_m) * std::cos(latitude) * std::sin(longitude),
          (radius * (1.0 - kEccentricitySquared) + altitude_m) * sin_latitude};
}

// 把相机安装方向和原点位置写成 IMU 坐标变换。
Eigen::Matrix4d ImuFromCamera(const YAML::Node& mount) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  const YAML::Node axes = mount["camera0_axes_in_imu"];
  transform.block<3, 1>(0, 0) = ReadVector3(axes["x"]);
  transform.block<3, 1>(0, 1) = ReadVector3(axes["y"]);
  transform.block<3, 1>(0, 2) = ReadVector3(axes["z"]);
  transform.block<3, 1>(0, 3) = ReadVector3(mount["camera0_origin_in_imu_m"]);
  return transform;
}

// 以第一台雷达为原点装载雷达、相机和天线外参。
void LoadSensorGeometry(GeometryConfig* config, const YAML::Node& mount,
                        const YAML::Node& rig, const YAML::Node& antenna) {
  const Eigen::Matrix4d imu_from_camera = ImuFromCamera(mount);
  const Eigen::Matrix4d first_lidar =
      imu_from_camera *
      ReadMatrix4(rig["lidars"]["0"]["T_reference_from_lidar"]);
  const Eigen::Vector3d origin = first_lidar.block<3, 1>(0, 3);
  for (size_t index = 0; index < config->reference_from_lidar.size(); ++index) {
    const std::string key = std::to_string(index);
    auto& transform = config->reference_from_lidar[index];
    transform = imu_from_camera *
                ReadMatrix4(rig["lidars"][key]["T_reference_from_lidar"]);
    transform.block<3, 1>(0, 3) -= origin;
  }
  config->reference_from_camera0 =
      imu_from_camera *
      ReadMatrix4(rig["cameras"]["0"]["T_reference_from_camera"]);
  config->reference_from_camera0.block<3, 1>(0, 3) -= origin;
  config->antenna_in_camera0 =
      ReadVector3(antenna["antenna_origin_in_camera0_m"]);
}

// 读取陀螺仪偏置和建图时计算的 GNSS 对齐矩阵。
void LoadGnssGeometry(GeometryConfig* config, const YAML::Node& bias,
                      const YAML::Node& alignment) {
  config->gyro_bias = ReadVector3(bias["gyro_bias_rad_s"]);
  config->ecef_origin = ReadVector3(alignment["ecef_origin"]);
  config->ecef_to_enu = ReadMatrix3(alignment["ecef_to_enu"]);
  config->enu_to_map = ReadMatrix3(alignment["enu_to_map_rotation"]);
  config->enu_to_map_translation =
      ReadVector3(alignment["enu_to_map_translation"]);
}

}  // namespace

// 加载校准文件，并构造雷达、天线及 GNSS 坐标变换。
GeometryConfig GeometryConfig::Load(
    const std::filesystem::path& calibration_dir,
    const std::filesystem::path& map_run,
    const std::filesystem::path& gyro_bias_file) {
  const YAML::Node mount =
      YAML::LoadFile((calibration_dir / "config/imu_mount.json").string());
  const YAML::Node rig = YAML::LoadFile(
      (calibration_dir / "results/rig_cam0_with_lidars_paired_1203.json")
          .string());
  const YAML::Node antenna =
      YAML::LoadFile((calibration_dir / "config/antenna_mount.json").string());
  const YAML::Node bias = YAML::LoadFile(gyro_bias_file.string());
  const YAML::Node alignment =
      YAML::LoadFile((map_run / "gnss_alignment.json").string());
  GeometryConfig config;
  LoadSensorGeometry(&config, mount, rig, antenna);
  LoadGnssGeometry(&config, bias, alignment);
  return config;
}

// 完成经纬高到 ECEF、ENU 和地图坐标系的变换。
Eigen::Vector3d GeometryConfig::GnssToMap(double longitude_deg,
                                          double latitude_deg,
                                          double altitude_m) const {
  const Eigen::Vector3d ecef =
      GeodeticToEcef(longitude_deg, latitude_deg, altitude_m);
  const Eigen::Vector3d enu = ecef_to_enu * (ecef - ecef_origin);
  return enu_to_map * enu + enu_to_map_translation;
}

// 由相机到天线的外参求出参考坐标系下的天线杆臂。
Eigen::Vector3d GeometryConfig::AntennaLeverArm() const {
  return reference_from_camera0.block<3, 3>(0, 0) * antenna_in_camera0 +
         reference_from_camera0.block<3, 1>(0, 3);
}

// 读取 SLAM 保存的位姿点云并验证时间戳严格递增。
PoseTimeline::PoseTimeline(const std::filesystem::path& pcd_file) {
  if (pcd_file.extension() == ".csv") {
    LoadCsv(pcd_file);
    return;
  }
  pcl::PCLPointCloud2 cloud;
  if (pcl::io::loadPCDFile(pcd_file.string(), cloud) < 0 || cloud.width == 0) {
    throw std::runtime_error("无法读取 SLAM 位姿文件：" + pcd_file.string());
  }
  const auto& x = FindField(cloud, "x");
  const auto& y = FindField(cloud, "y");
  const auto& z = FindField(cloud, "z");
  const auto& roll = FindField(cloud, "roll");
  const auto& pitch = FindField(cloud, "pitch");
  const auto& yaw = FindField(cloud, "yaw");
  const auto& time = FindField(cloud, "time");
  for (uint32_t index = 0; index < cloud.width * cloud.height; ++index) {
    Pose3d pose;
    pose.position = {ReadPcdValue(cloud, index, x),
                     ReadPcdValue(cloud, index, y),
                     ReadPcdValue(cloud, index, z)};
    const Eigen::AngleAxisd rotation_z(ReadPcdValue(cloud, index, yaw),
                                       Eigen::Vector3d::UnitZ());
    const Eigen::AngleAxisd rotation_y(ReadPcdValue(cloud, index, pitch),
                                       Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd rotation_x(ReadPcdValue(cloud, index, roll),
                                       Eigen::Vector3d::UnitX());
    pose.orientation = Eigen::Quaterniond(rotation_z * rotation_y * rotation_x);
    const double stamp = ReadPcdValue(cloud, index, time);
    Append(stamp, pose);
  }
}

// 读取建图节点导出的连续位姿，四元数按 x、y、z、w 排列。
void PoseTimeline::LoadCsv(const std::filesystem::path& file) {
  std::ifstream input(file);
  std::string line;
  if (!std::getline(input, line) ||
      line.substr(0, line.find('\r')) != "stamp_s,x,y,z,qx,qy,qz,qw") {
    throw std::runtime_error("无法读取建图轨迹表头：" + file.string());
  }
  while (std::getline(input, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    Pose3d pose;
    auto& q = pose.orientation;
    auto& p = pose.position;
    double stamp;
    if (!(row >> stamp >> p.x() >> p.y() >> p.z() >> q.x() >> q.y() >> q.z() >>
          q.w())) {
      throw std::runtime_error("建图连续轨迹行格式无效");
    }
    Append(stamp, pose);
  }
}

// 同时检查两种建图文件中的时间戳、位置和四元数。
void PoseTimeline::Append(double stamp, const Pose3d& pose) {
  if (!std::isfinite(stamp) || (!stamps_.empty() && stamp <= stamps_.back()) ||
      !pose.position.allFinite() || !pose.orientation.coeffs().allFinite() ||
      pose.orientation.squaredNorm() < 1e-12) {
    throw std::runtime_error("建图轨迹位姿或时间戳无效");
  }
  stamps_.push_back(stamp);
  poses_.push_back({pose.position, pose.orientation.normalized()});
}

// 在 SLAM 轨迹中插值单帧初始位姿，不参与后续定位更新。
Pose3d PoseTimeline::At(double stamp) const {
  if (stamps_.empty() || stamp < stamps_.front() - 0.02 ||
      stamp > stamps_.back() + 0.02) {
    throw std::out_of_range("定位首帧超出 SLAM 位姿时间范围");
  }
  stamp = std::clamp(stamp, stamps_.front(), stamps_.back());
  const auto upper = std::lower_bound(stamps_.begin(), stamps_.end(), stamp);
  if (upper == stamps_.begin()) return poses_.front();
  if (upper == stamps_.end()) return poses_.back();
  const size_t right = static_cast<size_t>(upper - stamps_.begin());
  const size_t left = right - 1;
  const double alpha =
      (stamp - stamps_[left]) / (stamps_[right] - stamps_[left]);
  Pose3d pose;
  pose.position =
      poses_[left].position * (1.0 - alpha) + poses_[right].position * alpha;
  pose.orientation =
      poses_[left].orientation.slerp(alpha, poses_[right].orientation);
  return pose;
}

// 按时间顺序累积去偏角速度，得到相对起始时刻的旋转。
void ImuHistory::Add(double stamp, const Eigen::Vector3d& angular_velocity) {
  if (!samples_.empty() && stamp <= samples_.back().stamp) return;
  ImuSample sample;
  sample.stamp = stamp;
  sample.angular_velocity = angular_velocity;
  if (!samples_.empty()) {
    const ImuSample& previous = samples_.back();
    const double interval = stamp - previous.stamp;
    const Eigen::Vector3d average =
        0.5 * (previous.angular_velocity + angular_velocity);
    const double angle = average.norm() * interval;
    sample.integrated_rotation = previous.integrated_rotation;
    if (angle > 1e-12) {
      sample.integrated_rotation *=
          Eigen::Quaterniond(Eigen::AngleAxisd(angle, average.normalized()));
      sample.integrated_rotation.normalize();
    }
  }
  samples_.push_back(sample);
}

// 确认 IMU 缓存覆盖整帧扫描的点采样时间。
bool ImuHistory::Covers(double start_stamp, double end_stamp) const {
  return samples_.size() >= 2 && samples_.front().stamp <= start_stamp &&
         samples_.back().stamp >= end_stamp;
}

// IMU 缓存按时间递增，早于左边界的扫描无法补齐。
bool ImuHistory::StartsAfter(double stamp) const {
  return !samples_.empty() && samples_.front().stamp > stamp;
}

// 对目标时刻两侧的累计姿态做球面线性插值。
Eigen::Quaterniond ImuHistory::RotationAt(double stamp) const {
  const auto upper =
      std::lower_bound(samples_.begin(), samples_.end(), stamp,
                       [](const ImuSample& sample, double value) {
                         return sample.stamp < value;
                       });
  if (upper == samples_.begin()) return upper->integrated_rotation;
  if (upper == samples_.end()) return samples_.back().integrated_rotation;
  const ImuSample& right = *upper;
  const ImuSample& left = *(upper - 1);
  const double alpha = (stamp - left.stamp) / (right.stamp - left.stamp);
  return left.integrated_rotation.slerp(alpha, right.integrated_rotation);
}

// 裁掉过旧 IMU 样本，同时保留一个姿态插值左边界。
void ImuHistory::TrimBefore(double stamp) {
  while (samples_.size() > 2 && samples_[1].stamp < stamp) {
    samples_.pop_front();
  }
}

// 把 ROS 秒和纳秒字段转换为秒级双精度时间戳。
double StampSeconds(int32_t sec, uint32_t nanosec) {
  return static_cast<double>(sec) + static_cast<double>(nanosec) * 1e-9;
}

}  // namespace ndt_localization
