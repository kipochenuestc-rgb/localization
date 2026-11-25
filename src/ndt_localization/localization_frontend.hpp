#ifndef NDT_LOCALIZATION__LOCALIZATION_FRONTEND_HPP_
#define NDT_LOCALIZATION__LOCALIZATION_FRONTEND_HPP_

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_epsilon_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <array>
#include <deque>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

#include "localization_geometry.hpp"

namespace ndt_localization {

class LocalizationFrontend : public rclcpp::Node {
 public:
  // 加载配置并连接原始雷达、IMU、GNSS 和 EKF 话题。
  LocalizationFrontend();

 private:
  using Cloud = sensor_msgs::msg::PointCloud2;
  using CloudConstPtr = Cloud::ConstSharedPtr;
  using SyncPolicy =
      message_filters::sync_policies::ApproximateEpsilonTime<Cloud, Cloud,
                                                             Cloud, Cloud>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  struct PendingScan {
    std::array<CloudConstPtr, 4> lidars;
    double stamp = 0.0;
    double start_stamp = 0.0;
    double end_stamp = 0.0;
  };

  // 同步四路雷达消息并排队等待对应的陀螺仪时间范围。
  void OnClouds(const CloudConstPtr& lidar1, const CloudConstPtr& lidar2,
                const CloudConstPtr& lidar3, const CloudConstPtr& lidar4);

  // 去除陀螺仪偏置、发布 EKF 输入并推进待处理扫描。
  void OnImu(sensor_msgs::msg::Imu::ConstSharedPtr message);

  // 在 IMU 数据覆盖完整扫描后执行旋转去畸变和体素滤波。
  void ProcessReadyScans();

  // 读取 ROS 参数并计算天线杆臂。
  void LoadParameters();

  // 创建前端发布器和轨迹发布器。
  void CreatePublishers();

  // 订阅 IMU、GNSS、NDT 以及 EKF 输出。
  void CreateSubscriptions();

  // 连接四路雷达并建立近似时间同步器。
  void ConnectLidars();

  // 对一组同步雷达点云完成拼接和发布。
  void ProcessScan(const PendingScan& scan);

  // 把体素滤波后的扫描发布给 NDT 定位器。
  void PublishScan(const PendingScan& scan,
                   const pcl::PointCloud<pcl::PointXYZ>& cloud);

  // 把单路雷达点变换到扫描时刻的统一参考坐标系。
  void AppendDeskewedCloud(const Cloud& message, size_t lidar_index,
                           const Eigen::Quaterniond& scan_rotation,
                           pcl::PointCloud<pcl::PointXYZ>& output);

  // 检查单点范围、采样时间及坐标是否有效。
  bool IsUsablePoint(const pcl::PointXYZ& point, double relative_time) const;

  // 把单点旋转到扫描参考时刻并变换到统一坐标系。
  Eigen::Vector3d DeskewPoint(const pcl::PointXYZ& point, double point_stamp,
                              const Eigen::Quaterniond& scan_rotation,
                              size_t lidar_index) const;

  // 将 GNSS 天线位置换算到地图系并发布给 EKF。
  void OnGnss(sensor_msgs::msg::NavSatFix::ConstSharedPtr message);

  // 判断 GNSS 修复状态、经纬高和时间戳是否有效。
  bool IsValidFix(const sensor_msgs::msg::NavSatFix& fix) const;

  // 把原始天线位置追加到 RViz 的 GNSS 路径。
  void AppendGnssPath(const sensor_msgs::msg::NavSatFix& fix,
                      const Eigen::Vector3d& antenna_position);

  // 缓存最新 NDT 估计，供 GNSS 杆臂补偿和创新门限使用。
  void OnNdt(nav_msgs::msg::Odometry::ConstSharedPtr message);

  // 缓存 EKF 输出并按时间间隔更新 RViz 定位轨迹。
  void OnEkf(nav_msgs::msg::Odometry::ConstSharedPtr message);

  // 定时发布累计轨迹，保证回放完成后 RViz 仍可见。
  void PublishPaths();

  // 选择时间最新的滤波位姿或 NDT 位姿用于天线杆臂补偿。
  bool LatestPose(Pose3d* pose) const;

  // 把 NavSatFix 的 ENU 协方差旋转到地图坐标系。
  Eigen::Matrix3d GnssCovariance(const sensor_msgs::msg::NavSatFix& fix) const;

  // 发布 GNSS 天线轨迹以及已补偿杆臂的 EKF 测量。
  void PublishGnssMeasurement(const sensor_msgs::msg::NavSatFix& fix,
                              const Eigen::Vector3d& antenna_position,
                              const Pose3d& pose);

  GeometryConfig geometry_;
  ImuHistory imu_history_;
  Eigen::Vector3d antenna_lever_arm_ = Eigen::Vector3d::Zero();
  std::deque<PendingScan> pending_scans_;
  std::array<message_filters::Subscriber<Cloud>, 4> lidar_subscribers_;
  std::unique_ptr<Synchronizer> synchronizer_;
  bool have_ndt_ = false;
  bool have_ekf_ = false;
  Pose3d latest_ndt_pose_;
  Pose3d latest_ekf_pose_;
  double latest_ndt_stamp_ = 0.0;
  double latest_ekf_stamp_ = 0.0;
  double last_path_stamp_ = 0.0;
  double max_point_time_s_ = 0.15;
  double max_gnss_innovation_m_ = 3.0;
  bool use_gnss_ = true;
  bool reliable_replay_ = false;
  nav_msgs::msg::Path trajectory_;
  nav_msgs::msg::Path gnss_path_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr gnss_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr gnss_path_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gnss_subscriber_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ndt_subscriber_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ekf_subscriber_;
  rclcpp::TimerBase::SharedPtr path_timer_;
};

}  // namespace ndt_localization

#endif  // NDT_LOCALIZATION__LOCALIZATION_FRONTEND_HPP_
