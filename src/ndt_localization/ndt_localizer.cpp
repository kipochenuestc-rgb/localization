#include "ndt_localizer.hpp"

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ndt_localization {
namespace {

// 将 SLAM 位姿转换成 NDT 使用的齐次变换矩阵。
Eigen::Matrix4f PoseMatrix(const Pose3d& pose) {
  Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
  matrix.block<3, 3>(0, 0) = pose.orientation.cast<float>().toRotationMatrix();
  matrix.block<3, 1>(0, 3) = pose.position.cast<float>();
  return matrix;
}

// 由位置和欧拉角构造没有地图先验时使用的初始位姿。
Eigen::Matrix4f FallbackPose(const std::vector<double>& values) {
  const Eigen::AngleAxisf yaw(values[5], Eigen::Vector3f::UnitZ());
  const Eigen::AngleAxisf pitch(values[4], Eigen::Vector3f::UnitY());
  const Eigen::AngleAxisf roll(values[3], Eigen::Vector3f::UnitX());
  Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
  matrix.block<3, 3>(0, 0) = (yaw * pitch * roll).toRotationMatrix();
  matrix.block<3, 1>(0, 3) = Eigen::Vector3f(values[0], values[1], values[2]);
  return matrix;
}

// 计算两个旋转矩阵之间的最小角度差。
double RotationDifference(const Eigen::Matrix4f& left,
                          const Eigen::Matrix4f& right) {
  const Eigen::Matrix3f delta =
      left.block<3, 3>(0, 0).transpose() * right.block<3, 3>(0, 0);
  return Eigen::AngleAxisf(delta).angle();
}

}  // namespace

// 加载配置、地图和 NDT 优化器，并连接定位话题。
NdtLocalizer::NdtLocalizer() : Node("ndt_map_localizer") {
  LoadParameters();
  LoadMap();
  ConfigureRegistration();
  CreatePublishers();
  CreateSubscriptions();
  map_timer_ = create_wall_timer(std::chrono::seconds(2),
                                 std::bind(&NdtLocalizer::PublishMap, this));
  PublishMap();
  RCLCPP_INFO(get_logger(), "NDT 地图已加载，共 %zu 个点", map_->size());
}

// 读取地图路径、初始化位姿和 NDT 质量门限。
void NdtLocalizer::LoadParameters() {
  map_file_ = declare_parameter<std::string>("map_file", "");
  pose_file_ = declare_parameter<std::string>("pose_file", "");
  trajectory_file_ = declare_parameter<std::string>("trajectory_file", "");
  fallback_initial_pose_ = declare_parameter<std::vector<double>>(
      "initial_pose", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  if (map_file_.empty() || fallback_initial_pose_.size() != 6) {
    throw std::invalid_argument("map_file 和六元素 initial_pose 必须有效");
  }
  fitness_limit_ = declare_parameter<double>("fitness_limit", 0.3);
  inlier_limit_ = declare_parameter<double>("inlier_limit", 0.75);
  translation_limit_ = declare_parameter<double>("translation_limit_m", 2.0);
  rotation_limit_ = declare_parameter<double>("rotation_limit_rad", 0.2);
  map_leaf_size_ = declare_parameter<double>("map_leaf_size_m", 0.4);
  scan_leaf_size_ = declare_parameter<double>("scan_leaf_size_m", 0.65);
  local_map_radius_m_ = declare_parameter<double>("local_map_radius_m", 90.0);
  max_iterations_ = declare_parameter<int>("max_iterations", 50);
}

// 设置 NDT 分布网格、优化步长和收敛条件。
void NdtLocalizer::ConfigureRegistration() {
  ndt_.setResolution(declare_parameter<double>("resolution", 1.0));
  ndt_.setStepSize(0.3);
  ndt_.setTransformationEpsilon(0.0001);
  ndt_.setTransformationRotationEpsilon(std::cos(0.001));
  ndt_.setMaximumIterations(max_iterations_);
}

// 读取完整建图点云并生成 NDT 使用的体素地图。
void NdtLocalizer::LoadMap() {
  Cloud::Ptr raw(new Cloud);
  if (pcl::io::loadPCDFile(map_file_, *raw) < 0 || raw->empty()) {
    throw std::runtime_error("无法读取定位地图：" + map_file_);
  }
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(raw);
  voxel.setLeafSize(map_leaf_size_, map_leaf_size_, map_leaf_size_);
  voxel.filter(*map_);
  if (map_->empty()) throw std::runtime_error("降采样后的地图为空");
  if (!pose_file_.empty())
    pose_timeline_ = std::make_unique<PoseTimeline>(pose_file_);
}

// 建立 NDT 地图及配准结果的 ROS 发布器。
void NdtLocalizer::CreatePublishers() {
  const auto map_qos = rclcpp::QoS(1).reliable().transient_local();
  map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/localization/map", map_qos);
  scan_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/localization/registered_scan", 10);
  pose_publisher_ =
      create_publisher<nav_msgs::msg::Odometry>("/localization/ndt", 10);
  status_publisher_ =
      create_publisher<std_msgs::msg::String>("/localization/ndt_status", 10);
}

// 连接 EKF 预测位姿和四雷达融合点云输入。
void NdtLocalizer::CreateSubscriptions() {
  prediction_subscriber_ = create_subscription<nav_msgs::msg::Odometry>(
      "/localization/ekf", 100,
      std::bind(&NdtLocalizer::OnPrediction, this, std::placeholders::_1));
  scan_subscriber_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/localization/input_scan", rclcpp::QoS(20).reliable(),
      std::bind(&NdtLocalizer::OnScan, this, std::placeholders::_1));
}

// 缓存 EKF 当前位姿，为下一帧 NDT 配准提供初值。
void NdtLocalizer::OnPrediction(
    nav_msgs::msg::Odometry::ConstSharedPtr message) {
  const auto& position = message->pose.pose.position;
  const auto& orientation = message->pose.pose.orientation;
  const Eigen::Quaternionf rotation(orientation.w, orientation.x, orientation.y,
                                    orientation.z);
  prediction_.setIdentity();
  prediction_.block<3, 3>(0, 0) = rotation.normalized().toRotationMatrix();
  prediction_(0, 3) = static_cast<float>(position.x);
  prediction_(1, 3) = static_cast<float>(position.y);
  prediction_(2, 3) = static_cast<float>(position.z);
  prediction_time_ = rclcpp::Time(message->header.stamp);
}

// 从 SLAM 轨迹插值得到首次 NDT 配准的地图初值。
Eigen::Matrix4f NdtLocalizer::InitialPose(double stamp) const {
  if (pose_timeline_) {
    try {
      return PoseMatrix(pose_timeline_->At(stamp));
    } catch (const std::out_of_range&) {
      if (trajectory_file_.empty()) throw;
      RCLCPP_INFO(get_logger(), "首帧超出关键帧范围，使用建图连续轨迹初始化");
      return PoseMatrix(PoseTimeline(trajectory_file_).At(stamp));
    }
  }
  return FallbackPose(fallback_initial_pose_);
}

// 优先采用同一回放时间附近的 EKF 预测位姿。
Eigen::Matrix4f NdtLocalizer::GuessPose(double stamp) {
  if (!seeded_) {
    last_pose_ = InitialPose(stamp);
    pose_timeline_.reset();
    seeded_ = true;
  }
  if (!initialized_) return last_pose_;
  const double prediction_stamp = prediction_time_.seconds();
  if (prediction_stamp > 0.0 && prediction_stamp <= stamp + 0.05 &&
      std::abs(stamp - prediction_stamp) < 0.5) {
    return prediction_;
  }
  Eigen::Matrix4f guess = last_pose_;
  if (have_velocity_ && stamp > last_stamp_ && stamp - last_stamp_ < 1.5) {
    guess.block<3, 1>(0, 3) +=
        velocity_ * static_cast<float>(stamp - last_stamp_);
  }
  return guess;
}

// 依照当前位姿从完整地图截取局部 NDT 目标点云。
void NdtLocalizer::UpdateTarget(const Eigen::Vector3f& center) {
  if ((center - target_center_).norm() <= 10.0F) return;
  target_->clear();
  target_center_ = center;
  const float radius_squared =
      static_cast<float>(local_map_radius_m_ * local_map_radius_m_);
  for (const auto& point : *map_) {
    if ((point.getVector3fMap() - center).squaredNorm() < radius_squared) {
      target_->push_back(point);
    }
  }
  ndt_.setInputTarget(target_);
  target_tree_.setInputCloud(target_);
}

// 解析并体素降采样一帧前端融合点云。
NdtLocalizer::Cloud::Ptr NdtLocalizer::FilterScan(
    const sensor_msgs::msg::PointCloud2& message) const {
  Cloud::Ptr raw(new Cloud);
  Cloud::Ptr filtered(new Cloud);
  pcl::fromROSMsg(message, *raw);
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(raw);
  voxel.setLeafSize(scan_leaf_size_, scan_leaf_size_, scan_leaf_size_);
  voxel.filter(*filtered);
  return filtered;
}

// 运行一次 NDT 优化并收集位置、角度和拟合质量。
NdtLocalizer::Registration NdtLocalizer::RegisterScan(
    const Cloud::ConstPtr& source, const Eigen::Matrix4f& guess) {
  const auto start = std::chrono::steady_clock::now();
  Registration result;
  result.pose = guess;
  ndt_.setInputSource(source);
  ndt_.align(*result.aligned, guess);
  result.pose = ndt_.getFinalTransformation();
  result.converged = ndt_.hasConverged();
  result.iterations = ndt_.getFinalNumIteration();
  result.fitness = ndt_.getFitnessScore(2.0);
  result.correction_m =
      (result.pose.block<3, 1>(0, 3) - guess.block<3, 1>(0, 3)).norm();
  result.correction_rad = RotationDifference(guess, result.pose);
  result.inlier_fraction = CalculateInlierFraction(*result.aligned);
  result.compute_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

// 抽样统计配准点云落在地图近邻范围内的比例。
double NdtLocalizer::CalculateInlierFraction(const Cloud& aligned) const {
  std::vector<int> indices(1);
  std::vector<float> distances(1);
  size_t hits = 0;
  size_t samples = 0;
  for (size_t index = 0; index < aligned.size(); index += 3) {
    ++samples;
    if (target_tree_.nearestKSearch(aligned[index], 1, indices, distances) >
            0 &&
        distances[0] < 0.64F) {
      ++hits;
    }
  }
  return samples == 0 ? 0.0 : static_cast<double>(hits) / samples;
}

// 按收敛状态和地图匹配质量过滤异常位姿。
bool NdtLocalizer::IsAccepted(const Registration& result) const {
  return result.converged && result.iterations < max_iterations_ &&
         result.pose.allFinite() && std::isfinite(result.fitness) &&
         result.fitness < fitness_limit_ &&
         result.inlier_fraction > inlier_limit_ &&
         result.correction_m < translation_limit_ &&
         result.correction_rad < rotation_limit_;
}

// 登记有效 NDT 结果并更新下一帧使用的速度预测。
void NdtLocalizer::UpdateMotion(const Registration& result, double stamp) {
  const double interval = stamp - last_stamp_;
  if (initialized_ && interval > 0.0 && interval < 1.5) {
    velocity_ = (result.pose.block<3, 1>(0, 3) - last_pose_.block<3, 1>(0, 3)) /
                static_cast<float>(interval);
    have_velocity_ = velocity_.allFinite() && velocity_.norm() < 25.0F;
  } else {
    have_velocity_ = false;
  }
  last_pose_ = result.pose;
  last_stamp_ = stamp;
  initialized_ = true;
}

// 把已接受的地图位姿和质量相关协方差写入里程计消息。
nav_msgs::msg::Odometry NdtLocalizer::MakeOdometry(
    const Registration& result,
    const sensor_msgs::msg::PointCloud2& source) const {
  nav_msgs::msg::Odometry message;
  message.header = source.header;
  message.header.frame_id = "map";
  message.child_frame_id = "localization_base";
  const auto& position = result.pose.block<3, 1>(0, 3);
  const Eigen::Quaternionf rotation(result.pose.block<3, 3>(0, 0));
  message.pose.pose.position.x = position.x();
  message.pose.pose.position.y = position.y();
  message.pose.pose.position.z = position.z();
  message.pose.pose.orientation.x = rotation.x();
  message.pose.pose.orientation.y = rotation.y();
  message.pose.pose.orientation.z = rotation.z();
  message.pose.pose.orientation.w = rotation.w();
  const double variance = std::clamp(result.fitness * 0.2, 0.01, 0.25);
  for (int index = 0; index < 3; ++index) {
    message.pose.covariance[index * 7] = variance;
    message.pose.covariance[(index + 3) * 7] = 0.0012;
  }
  return message;
}

// 发布 NDT 位姿测量和配准后扫描供 EKF 与 RViz 使用。
void NdtLocalizer::PublishRegistration(
    const Registration& result, const sensor_msgs::msg::PointCloud2& source) {
  pose_publisher_->publish(MakeOdometry(result, source));
  sensor_msgs::msg::PointCloud2 registered;
  pcl::toROSMsg(*result.aligned, registered);
  registered.header = source.header;
  registered.header.frame_id = "map";
  scan_publisher_->publish(registered);
}

// 生成供日志和回放审核使用的单行质量记录。
std::string NdtLocalizer::FormatStatus(const Registration& result,
                                       double stamp) const {
  const Eigen::Quaternionf rotation(result.pose.block<3, 3>(0, 0));
  std::ostringstream output;
  output.precision(17);
  output << "{\"stamp_s\":" << stamp
         << ",\"accepted\":" << (result.accepted ? "true" : "false")
         << ",\"iterations\":" << result.iterations << ",\"pose\":["
         << result.pose(0, 3) << ',' << result.pose(1, 3) << ','
         << result.pose(2, 3) << ',' << rotation.x() << ',' << rotation.y()
         << ',' << rotation.z() << ',' << rotation.w() << ']'
         << ",\"fitness_m2\":" << result.fitness
         << ",\"inlier_fraction\":" << result.inlier_fraction
         << ",\"correction_m\":" << result.correction_m
         << ",\"correction_rad\":" << result.correction_rad
         << ",\"compute_s\":" << result.compute_s << '}';
  return output.str();
}

// 发布配准质量状态并同步写入 ROS 日志。
void NdtLocalizer::PublishStatus(const Registration& result, double stamp) {
  std_msgs::msg::String message;
  message.data = FormatStatus(result, stamp);
  status_publisher_->publish(message);
  RCLCPP_INFO(get_logger(), "%s", message.data.c_str());
}

// 周期性发布完整固定地图，方便 RViz 接入后立即显示。
void NdtLocalizer::PublishMap() {
  sensor_msgs::msg::PointCloud2 message;
  pcl::toROSMsg(*map_, message);
  message.header.frame_id = "map";
  message.header.stamp = now();
  map_publisher_->publish(message);
}

// 处理点云并在质量通过时更新 NDT 与 EKF 定位结果。
void NdtLocalizer::OnScan(
    sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
  const double stamp = rclcpp::Time(message->header.stamp).seconds();
  const Eigen::Matrix4f guess = GuessPose(stamp);
  UpdateTarget(guess.block<3, 1>(0, 3));
  const Cloud::Ptr source = FilterScan(*message);
  Registration result;
  result.pose = guess;
  if (source->size() > 100 && target_->size() > 1000) {
    result = RegisterScan(source, guess);
    result.accepted = IsAccepted(result);
  }
  if (result.accepted) {
    UpdateMotion(result, stamp);
    PublishRegistration(result, *message);
  }
  PublishStatus(result, stamp);
}

}  // namespace ndt_localization

// 启动 NDT 定位节点并将初始化异常报告给 ROS 日志。
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<ndt_localization::NdtLocalizer>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("ndt_map_localizer"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
