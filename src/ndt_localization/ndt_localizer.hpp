#ifndef NDT_LOCALIZATION__NDT_LOCALIZER_HPP_
#define NDT_LOCALIZATION__NDT_LOCALIZER_HPP_

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>

#include <Eigen/Core>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>
#include <vector>

#include "localization_geometry.hpp"

namespace ndt_localization {

class NdtLocalizer : public rclcpp::Node {
 public:
  // 加载地图、初始化 NDT 并创建定位话题。
  NdtLocalizer();

 private:
  using Cloud = pcl::PointCloud<pcl::PointXYZ>;

  struct Registration {
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    Cloud::Ptr aligned{new Cloud};
    double fitness = 1e6;
    double inlier_fraction = 0.0;
    double correction_m = 1e6;
    double correction_rad = 1e6;
    double compute_s = 0.0;
    int iterations = 0;
    bool converged = false;
    bool accepted = false;
  };

  // 从 ROS 参数读取地图、位姿轨迹和 NDT 质量门限。
  void LoadParameters();

  // 建立地图点云降采样器和 NDT 优化器参数。
  void ConfigureRegistration();

  // 创建地图、点云、位姿和质量状态发布器。
  void CreatePublishers();

  // 订阅 EKF 预测位姿和前端拼接点云。
  void CreateSubscriptions();

  // 读取完整地图并建立降采样后的 NDT 目标云。
  void LoadMap();

  // 根据当前 EKF 输出更新 NDT 初值。
  void OnPrediction(nav_msgs::msg::Odometry::ConstSharedPtr message);

  // 处理一帧拼接雷达扫描并发布配准结果。
  void OnScan(sensor_msgs::msg::PointCloud2::ConstSharedPtr message);

  // 获取首帧 SLAM 初值或下一帧 EKF 预测位姿。
  Eigen::Matrix4f InitialPose(double stamp) const;

  // 选择当前扫描的 NDT 初值并应用速度外推。
  Eigen::Matrix4f GuessPose(double stamp);

  // 按预测位置裁取完整地图中的局部 NDT 子图。
  void UpdateTarget(const Eigen::Vector3f& center);

  // 降采样输入扫描，控制 NDT 的计算量。
  Cloud::Ptr FilterScan(const sensor_msgs::msg::PointCloud2& message) const;

  // 运行 NDT 并计算拟合误差和地图内点比例。
  Registration RegisterScan(const Cloud::ConstPtr& source,
                            const Eigen::Matrix4f& guess);

  // 统计配准点到局部地图的近邻比例。
  double CalculateInlierFraction(const Cloud& aligned) const;

  // 依据收敛状态、拟合误差和位姿跳变判定结果。
  bool IsAccepted(const Registration& result) const;

  // 保存当前有效位姿并更新短时速度外推量。
  void UpdateMotion(const Registration& result, double stamp);

  // 发布 NDT 位姿测量和配准后的点云。
  void PublishRegistration(const Registration& result,
                           const sensor_msgs::msg::PointCloud2& source);

  // 定时发布完整地图的降采样显示副本。
  void PublishMap();

  // 记录本帧 NDT 质量指标，供日志和监视器读取。
  void PublishStatus(const Registration& result, double stamp);

  // 把配准结果格式化为可审计的单行 JSON。
  std::string FormatStatus(const Registration& result, double stamp) const;

  // 把矩阵位姿转成地图系里程计测量及其协方差。
  nav_msgs::msg::Odometry MakeOdometry(
      const Registration& result,
      const sensor_msgs::msg::PointCloud2& source) const;

  Cloud::Ptr map_{new Cloud};
  Cloud::Ptr target_{new Cloud};
  pcl::KdTreeFLANN<pcl::PointXYZ> target_tree_;
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_;
  std::unique_ptr<PoseTimeline> pose_timeline_;
  std::string map_file_;
  std::string pose_file_;
  std::string trajectory_file_;
  std::vector<double> fallback_initial_pose_;
  Eigen::Matrix4f last_pose_ = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f prediction_ = Eigen::Matrix4f::Identity();
  Eigen::Vector3f target_center_ = Eigen::Vector3f::Constant(1e9F);
  Eigen::Vector3f velocity_ = Eigen::Vector3f::Zero();
  rclcpp::Time prediction_time_{0, 0, RCL_ROS_TIME};
  double last_stamp_ = 0.0;
  double fitness_limit_ = 0.3;
  double inlier_limit_ = 0.75;
  double translation_limit_ = 2.0;
  double rotation_limit_ = 0.2;
  double map_leaf_size_ = 0.4;
  double scan_leaf_size_ = 0.65;
  double local_map_radius_m_ = 90.0;
  bool have_velocity_ = false;
  bool seeded_ = false;
  bool initialized_ = false;
  int max_iterations_ = 50;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pose_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      scan_subscriber_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      prediction_subscriber_;
  rclcpp::TimerBase::SharedPtr map_timer_;
};

}  // namespace ndt_localization

#endif  // NDT_LOCALIZATION__NDT_LOCALIZER_HPP_
