#include "localization_frontend.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rmw/qos_profiles.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ndt_localization {
namespace {

// 查找 PointCloud2 中指定名称的字段。
const sensor_msgs::msg::PointField& FindField(
    const sensor_msgs::msg::PointCloud2& cloud, const std::string& name) {
  for (const auto& field : cloud.fields) {
    if (field.name == name) return field;
  }
  throw std::runtime_error("点云缺少字段：" + name);
}

// 读取当前小端点云中的浮点或无符号数值字段。
double ReadField(const sensor_msgs::msg::PointCloud2& cloud, size_t index,
                 const sensor_msgs::msg::PointField& field) {
  if (cloud.is_bigendian) throw std::runtime_error("暂不支持大端点云");
  if (cloud.width == 0 || field.count != 1) {
    throw std::runtime_error("点云宽度或字段数量无效");
  }
  const size_t row = index / cloud.width;
  const size_t column = index % cloud.width;
  const size_t offset =
      row * cloud.row_step + column * cloud.point_step + field.offset;
  if (offset + sizeof(uint32_t) > cloud.data.size()) {
    throw std::runtime_error("点云字段超出消息数据范围");
  }
  const uint8_t* data = cloud.data.data() + offset;
  if (field.datatype == sensor_msgs::msg::PointField::FLOAT32) {
    float value;
    std::memcpy(&value, data, sizeof(value));
    return value;
  }
  if (field.datatype == sensor_msgs::msg::PointField::UINT32) {
    uint32_t value;
    std::memcpy(&value, data, sizeof(value));
    return value;
  }
  throw std::runtime_error("点云字段类型不受支持：" + field.name);
}

// 从里程计消息中提取位置和方向。
Pose3d ReadPose(const nav_msgs::msg::Odometry& odometry) {
  Pose3d pose;
  const auto& position = odometry.pose.pose.position;
  const auto& orientation = odometry.pose.pose.orientation;
  pose.position = {position.x, position.y, position.z};
  pose.orientation = Eigen::Quaterniond(orientation.w, orientation.x,
                                        orientation.y, orientation.z);
  pose.orientation.normalize();
  return pose;
}

// 将 Eigen 三维向量写入 ROS 几何消息。
void WriteVector(const Eigen::Vector3d& value,
                 geometry_msgs::msg::Point* point) {
  point->x = value.x();
  point->y = value.y();
  point->z = value.z();
}

}  // namespace

// 创建 C++ 定位前端并初始化 ROS 话题连接。
LocalizationFrontend::LocalizationFrontend() : Node("localization_frontend") {
  LoadParameters();
  CreatePublishers();
  CreateSubscriptions();
  ConnectLidars();
  path_timer_ =
      create_wall_timer(std::chrono::milliseconds(500),
                        std::bind(&LocalizationFrontend::PublishPaths, this));
}

// 读取校准路径、地图对齐和定位门限参数。
void LocalizationFrontend::LoadParameters() {
  const std::string calibration =
      declare_parameter<std::string>("calibration_dir", "");
  const std::string map_run = declare_parameter<std::string>("map_run", "");
  const std::string bias = declare_parameter<std::string>("gyro_bias_file", "");
  if (calibration.empty() || map_run.empty() || bias.empty()) {
    throw std::invalid_argument(
        "必须提供 calibration_dir、map_run 和 gyro_bias_file");
  }
  geometry_ = GeometryConfig::Load(calibration, map_run, bias);
  antenna_lever_arm_ = geometry_.AntennaLeverArm();
  max_point_time_s_ = declare_parameter<double>("max_point_time_s", 0.15);
  max_gnss_innovation_m_ =
      declare_parameter<double>("max_gnss_innovation_m", 3.0);
  use_gnss_ = declare_parameter<bool>("use_gnss", true);
  reliable_replay_ = declare_parameter<bool>("reliable_replay", false);
}

// 创建 NDT、IMU、GNSS 和 RViz 路径发布器。
void LocalizationFrontend::CreatePublishers() {
  scan_publisher_ = create_publisher<Cloud>("/localization/input_scan", 20);
  imu_publisher_ =
      create_publisher<sensor_msgs::msg::Imu>("/localization/imu", 2000);
  gnss_publisher_ =
      create_publisher<nav_msgs::msg::Odometry>("/localization/gnss", 20);
  path_publisher_ =
      create_publisher<nav_msgs::msg::Path>("/localization/path", 10);
  gnss_path_publisher_ =
      create_publisher<nav_msgs::msg::Path>("/localization/gnss_path", 10);
  trajectory_.header.frame_id = "map";
  gnss_path_.header.frame_id = "map";
}

// 订阅滤波器输出和原始 IMU、GNSS 传感器数据。
void LocalizationFrontend::CreateSubscriptions() {
  auto imu_qos = rclcpp::SensorDataQoS().keep_last(512);
  if (reliable_replay_) imu_qos.reliable();
  imu_subscriber_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu/data", imu_qos,
      std::bind(&LocalizationFrontend::OnImu, this, std::placeholders::_1));
  gnss_subscriber_ = create_subscription<sensor_msgs::msg::NavSatFix>(
      "/gnss/fix_fixed", rclcpp::SensorDataQoS(),
      std::bind(&LocalizationFrontend::OnGnss, this, std::placeholders::_1));
  ndt_subscriber_ = create_subscription<nav_msgs::msg::Odometry>(
      "/localization/ndt", 20,
      std::bind(&LocalizationFrontend::OnNdt, this, std::placeholders::_1));
  ekf_subscriber_ = create_subscription<nav_msgs::msg::Odometry>(
      "/localization/ekf", 100,
      std::bind(&LocalizationFrontend::OnEkf, this, std::placeholders::_1));
}

// 连接四路雷达话题并限制同一组扫描的时间差。
void LocalizationFrontend::ConnectLidars() {
  const std::vector<std::string> topics =
      declare_parameter<std::vector<std::string>>(
          "lidar_topics", {"/point_cloud_hasco1", "/point_cloud_hasco2",
                           "/point_cloud_hasco3", "/point_cloud_hasco4"});
  if (topics.size() != lidar_subscribers_.size()) {
    throw std::invalid_argument("lidar_topics 必须包含四个话题");
  }
  auto qos = rclcpp::SensorDataQoS().keep_last(20);
  if (reliable_replay_) qos.reliable();
  for (size_t index = 0; index < lidar_subscribers_.size(); ++index) {
    lidar_subscribers_[index].subscribe(this, topics[index],
                                        qos.get_rmw_qos_profile());
  }
  synchronizer_ = std::make_unique<Synchronizer>(
      SyncPolicy(20, rclcpp::Duration::from_seconds(0.02)),
      lidar_subscribers_[0], lidar_subscribers_[1], lidar_subscribers_[2],
      lidar_subscribers_[3]);
  synchronizer_->registerCallback(std::bind(
      &LocalizationFrontend::OnClouds, this, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
}

// 缓存一组近似同步的雷达消息，等待 IMU 覆盖完整帧时间。
void LocalizationFrontend::OnClouds(const CloudConstPtr& lidar1,
                                    const CloudConstPtr& lidar2,
                                    const CloudConstPtr& lidar3,
                                    const CloudConstPtr& lidar4) {
  PendingScan scan{{lidar1, lidar2, lidar3, lidar4}};
  scan.stamp =
      StampSeconds(lidar1->header.stamp.sec, lidar1->header.stamp.nanosec);
  scan.start_stamp = scan.stamp;
  scan.end_stamp = scan.stamp + max_point_time_s_;
  for (const auto& lidar : scan.lidars) {
    const double lidar_stamp =
        StampSeconds(lidar->header.stamp.sec, lidar->header.stamp.nanosec);
    scan.start_stamp = std::min(scan.start_stamp, lidar_stamp);
    scan.end_stamp = std::max(scan.end_stamp, lidar_stamp + max_point_time_s_);
  }
  pending_scans_.push_back(std::move(scan));
  if (pending_scans_.size() > 8) {
    RCLCPP_WARN(get_logger(), "IMU 延迟过大，丢弃最早的一组雷达扫描");
    pending_scans_.pop_front();
  }
  ProcessReadyScans();
}

// 扣除陀螺仪偏置并发布 EKF 输入，同时处理就绪扫描。
void LocalizationFrontend::OnImu(
    sensor_msgs::msg::Imu::ConstSharedPtr message) {
  const double stamp =
      StampSeconds(message->header.stamp.sec, message->header.stamp.nanosec);
  const Eigen::Vector3d angular_velocity(
      message->angular_velocity.x - geometry_.gyro_bias.x(),
      message->angular_velocity.y - geometry_.gyro_bias.y(),
      message->angular_velocity.z - geometry_.gyro_bias.z());
  imu_history_.Add(stamp, angular_velocity);
  imu_history_.TrimBefore(stamp - 3.0);
  sensor_msgs::msg::Imu corrected = *message;
  corrected.header.frame_id = "localization_base";
  corrected.angular_velocity.x = angular_velocity.x();
  corrected.angular_velocity.y = angular_velocity.y();
  corrected.angular_velocity.z = angular_velocity.z();
  corrected.orientation_covariance[0] = -1.0;
  corrected.linear_acceleration_covariance[0] = -1.0;
  corrected.angular_velocity_covariance[0] = 1e-4;
  corrected.angular_velocity_covariance[4] = 1e-4;
  corrected.angular_velocity_covariance[8] = 1e-4;
  // 先用地图位姿初始化 EKF，避免从地图中段启动时在原点生成轨迹。
  if (have_ndt_) imu_publisher_->publish(corrected);
  ProcessReadyScans();
}

// 按时间顺序处理已有充分 IMU 覆盖的待定位扫描。
void LocalizationFrontend::ProcessReadyScans() {
  while (!pending_scans_.empty()) {
    const PendingScan& scan = pending_scans_.front();
    if (imu_history_.StartsAfter(scan.start_stamp)) {
      pending_scans_.pop_front();
      continue;
    }
    if (!imu_history_.Covers(scan.start_stamp, scan.end_stamp)) break;
    const double processed_stamp = scan.stamp;
    ProcessScan(scan);
    pending_scans_.pop_front();
    imu_history_.TrimBefore(processed_stamp - 0.2);
  }
}

// 拼接四路去畸变点云，体素滤波后交给 NDT。
void LocalizationFrontend::ProcessScan(const PendingScan& scan) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr combined(
      new pcl::PointCloud<pcl::PointXYZ>());
  const Eigen::Quaterniond scan_rotation = imu_history_.RotationAt(scan.stamp);
  for (size_t index = 0; index < scan.lidars.size(); ++index) {
    AppendDeskewedCloud(*scan.lidars[index], index, scan_rotation, *combined);
  }
  if (combined->size() < 100) {
    RCLCPP_WARN(get_logger(), "同步扫描有效点不足，跳过本帧");
    return;
  }
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(combined);
  voxel.setLeafSize(0.3F, 0.3F, 0.3F);
  pcl::PointCloud<pcl::PointXYZ> filtered;
  voxel.filter(filtered);
  PublishScan(scan, filtered);
}

// 将滤波点云发布到 NDT，并保留 LiDAR 1 的原始时间戳。
void LocalizationFrontend::PublishScan(
    const PendingScan& scan, const pcl::PointCloud<pcl::PointXYZ>& cloud) {
  Cloud message;
  pcl::toROSMsg(cloud, message);
  message.header = scan.lidars[0]->header;
  message.header.frame_id = "localization_base";
  scan_publisher_->publish(message);
}

// 解析单路点云的字段并逐点执行范围筛选和去畸变。
void LocalizationFrontend::AppendDeskewedCloud(
    const Cloud& message, size_t lidar_index,
    const Eigen::Quaterniond& scan_rotation,
    pcl::PointCloud<pcl::PointXYZ>& output) {
  const auto& x_field = FindField(message, "x");
  const auto& y_field = FindField(message, "y");
  const auto& z_field = FindField(message, "z");
  const auto& time_field = FindField(message, "timestamp");
  const size_t count = static_cast<size_t>(message.width) * message.height;
  for (size_t index = 0; index < count; ++index) {
    const pcl::PointXYZ point(ReadField(message, index, x_field),
                              ReadField(message, index, y_field),
                              ReadField(message, index, z_field));
    const double relative = ReadField(message, index, time_field) * 1e-6;
    if (IsUsablePoint(point, relative)) {
      const double point_stamp =
          StampSeconds(message.header.stamp.sec, message.header.stamp.nanosec) +
          relative;
      const Eigen::Vector3d corrected =
          DeskewPoint(point, point_stamp, scan_rotation, lidar_index);
      output.emplace_back(corrected.x(), corrected.y(), corrected.z());
    }
  }
}

// 筛除无效坐标、近距离回波和扫描时间范围外的点。
bool LocalizationFrontend::IsUsablePoint(const pcl::PointXYZ& point,
                                         double relative_time) const {
  const double range =
      std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z) && range > 1.5 && range < 70.0 &&
         relative_time >= 0.0 && relative_time < max_point_time_s_;
}

// 用 IMU 姿态差补偿点时间，并应用雷达到参考坐标外参。
Eigen::Vector3d LocalizationFrontend::DeskewPoint(
    const pcl::PointXYZ& point, double point_stamp,
    const Eigen::Quaterniond& scan_rotation, size_t lidar_index) const {
  const Eigen::Vector3d lidar_point(point.x, point.y, point.z);
  const Eigen::Matrix4d& transform =
      geometry_.reference_from_lidar[lidar_index];
  const Eigen::Vector3d reference_point =
      transform.block<3, 3>(0, 0) * lidar_point + transform.block<3, 1>(0, 3);
  const Eigen::Quaterniond point_rotation =
      imu_history_.RotationAt(point_stamp);
  const Eigen::Quaterniond relative =
      scan_rotation.conjugate() * point_rotation;
  return relative * reference_point;
}

// 判断 GNSS 状态、经纬高数值和时间戳是否可用。
bool LocalizationFrontend::IsValidFix(
    const sensor_msgs::msg::NavSatFix& fix) const {
  return fix.status.status >= 0 && std::isfinite(fix.longitude) &&
         std::isfinite(fix.latitude) && std::isfinite(fix.altitude);
}

// 将有效 GNSS 天线位置变换到地图并执行创新门限。
void LocalizationFrontend::OnGnss(
    sensor_msgs::msg::NavSatFix::ConstSharedPtr message) {
  if (!IsValidFix(*message)) return;
  const Eigen::Vector3d antenna = geometry_.GnssToMap(
      message->longitude, message->latitude, message->altitude);
  AppendGnssPath(*message, antenna);
  if (!use_gnss_) return;
  Pose3d pose;
  if (!LatestPose(&pose)) return;
  const Eigen::Vector3d reference =
      antenna - pose.orientation * antenna_lever_arm_;
  if ((reference.head<2>() - pose.position.head<2>()).norm() >
      max_gnss_innovation_m_) {
    RCLCPP_WARN(get_logger(), "GNSS 水平创新超过 %.1f m，拒绝该测量",
                max_gnss_innovation_m_);
    return;
  }
  PublishGnssMeasurement(*message, antenna, pose);
}

// 将 GNSS 天线原始轨迹追加到地图系路径。
void LocalizationFrontend::AppendGnssPath(
    const sensor_msgs::msg::NavSatFix& fix,
    const Eigen::Vector3d& antenna_position) {
  geometry_msgs::msg::PoseStamped point;
  point.header = fix.header;
  point.header.frame_id = "map";
  WriteVector(antenna_position, &point.pose.position);
  point.pose.orientation.w = 1.0;
  gnss_path_.header = point.header;
  gnss_path_.poses.push_back(point);
  gnss_path_publisher_->publish(gnss_path_);
}

// 缓存 NDT 位姿，作为 EKF 初始化时的 GNSS 杆臂参考。
void LocalizationFrontend::OnNdt(
    nav_msgs::msg::Odometry::ConstSharedPtr message) {
  latest_ndt_pose_ = ReadPose(*message);
  latest_ndt_stamp_ =
      StampSeconds(message->header.stamp.sec, message->header.stamp.nanosec);
  have_ndt_ = true;
}

// 缓存 EKF 输出并按时间间隔采样定位轨迹。
void LocalizationFrontend::OnEkf(
    nav_msgs::msg::Odometry::ConstSharedPtr message) {
  latest_ekf_pose_ = ReadPose(*message);
  latest_ekf_stamp_ =
      StampSeconds(message->header.stamp.sec, message->header.stamp.nanosec);
  have_ekf_ = true;
  if (latest_ekf_stamp_ - last_path_stamp_ < 0.095) return;
  geometry_msgs::msg::PoseStamped point;
  point.header = message->header;
  point.header.frame_id = "map";
  point.pose = message->pose.pose;
  trajectory_.header = point.header;
  trajectory_.poses.push_back(point);
  last_path_stamp_ = latest_ekf_stamp_;
}

// 发布累计定位路径和 GNSS 路径。
void LocalizationFrontend::PublishPaths() {
  if (!trajectory_.poses.empty()) path_publisher_->publish(trajectory_);
  if (!gnss_path_.poses.empty()) gnss_path_publisher_->publish(gnss_path_);
}

// 优先使用 EKF 位姿，没有滤波输出时退回到最新 NDT 位姿。
bool LocalizationFrontend::LatestPose(Pose3d* pose) const {
  if (have_ekf_ && latest_ekf_stamp_ >= latest_ndt_stamp_) {
    *pose = latest_ekf_pose_;
    return true;
  }
  if (have_ndt_) {
    *pose = latest_ndt_pose_;
    return true;
  }
  return false;
}

// 旋转 GNSS 协方差并为未知或无效协方差设置合理默认值。
Eigen::Matrix3d LocalizationFrontend::GnssCovariance(
    const sensor_msgs::msg::NavSatFix& fix) const {
  Eigen::Matrix3d enu = Eigen::Matrix3d::Zero();
  bool valid = fix.position_covariance_type !=
               sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      enu(row, col) = fix.position_covariance[row * 3 + col];
    }
    valid = valid && std::isfinite(enu(row, row)) && enu(row, row) > 0.0;
  }
  if (!valid || !enu.allFinite())
    enu = Eigen::Vector3d(0.25, 0.25, 4.0).asDiagonal();
  return geometry_.enu_to_map * enu * geometry_.enu_to_map.transpose();
}

// 发布杆臂补偿后的 GNSS 里程计供 robot_localization EKF 融合。
void LocalizationFrontend::PublishGnssMeasurement(
    const sensor_msgs::msg::NavSatFix& fix,
    const Eigen::Vector3d& antenna_position, const Pose3d& pose) {
  nav_msgs::msg::Odometry measurement;
  measurement.header = fix.header;
  measurement.header.frame_id = "map";
  measurement.child_frame_id = "localization_base";
  const Eigen::Vector3d reference_position =
      antenna_position - pose.orientation * antenna_lever_arm_;
  WriteVector(reference_position, &measurement.pose.pose.position);
  measurement.pose.pose.orientation.w = 1.0;
  const Eigen::Matrix3d covariance = GnssCovariance(fix);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      measurement.pose.covariance[row * 6 + col] = covariance(row, col);
    }
  }
  gnss_publisher_->publish(measurement);
}

}  // namespace ndt_localization

// 启动定位前端 ROS 2 节点并处理异常退出。
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<ndt_localization::LocalizationFrontend>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("localization_frontend"), "%s",
                 error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
