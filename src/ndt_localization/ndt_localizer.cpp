#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Geometry>
#include <sstream>

using Cloud=pcl::PointCloud<pcl::PointXYZ>;
class Localizer:public rclcpp::Node {
  Cloud::Ptr map_{new Cloud}, target_{new Cloud};
  pcl::KdTreeFLANN<pcl::PointXYZ> target_tree_;
  Eigen::Vector3f target_center_=Eigen::Vector3f::Constant(1e9f), velocity_=Eigen::Vector3f::Zero();
  bool have_velocity_=false; int max_iterations_=50;
  pcl::NormalDistributionsTransform<pcl::PointXYZ,pcl::PointXYZ> ndt_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_,scan_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr prediction_sub_;
  rclcpp::TimerBase::SharedPtr map_timer_;
  Eigen::Matrix4f last_=Eigen::Matrix4f::Identity(), prediction_=Eigen::Matrix4f::Identity();
  rclcpp::Time prediction_time_{0,0,RCL_ROS_TIME}; double last_time_=0.; bool initialized_=false;
 public:
  Localizer():Node("ndt_map_localizer") {
    auto filename=declare_parameter<std::string>("map_file","");
    auto seed=declare_parameter<std::vector<double>>("initial_pose",{0.,0.,0.,0.,0.,0.});
    if(seed.size()!=6)throw std::runtime_error("initial_pose requires x y z roll pitch yaw");
    last_.block<3,3>(0,0)=(Eigen::AngleAxisf(seed[5],Eigen::Vector3f::UnitZ())*Eigen::AngleAxisf(seed[4],Eigen::Vector3f::UnitY())*Eigen::AngleAxisf(seed[3],Eigen::Vector3f::UnitX())).toRotationMatrix();
    for(int j=0;j<3;++j)last_(j,3)=seed[j];
    if(pcl::io::loadPCDFile(filename,*map_)<0)throw std::runtime_error("Cannot load map");
    pcl::VoxelGrid<pcl::PointXYZ> voxel; voxel.setInputCloud(map_);voxel.setLeafSize(.4,.4,.4);Cloud::Ptr reduced(new Cloud);voxel.filter(*reduced);map_=reduced;
    // PCL compares squared translation to epsilon. With step=.1 and epsilon=.01,
    // a full 10 cm optimizer step incorrectly satisfied the stopping condition.
    ndt_.setResolution(declare_parameter<double>("resolution",1.));
    ndt_.setStepSize(.3);ndt_.setTransformationEpsilon(.0001);
    ndt_.setTransformationRotationEpsilon(std::cos(.001));ndt_.setMaximumIterations(max_iterations_);
    map_pub_=create_publisher<sensor_msgs::msg::PointCloud2>("/localization/map",rclcpp::QoS(1).reliable().transient_local());
    scan_pub_=create_publisher<sensor_msgs::msg::PointCloud2>("/localization/registered_scan",10);
    pose_pub_=create_publisher<nav_msgs::msg::Odometry>("/localization/ndt",10);status_pub_=create_publisher<std_msgs::msg::String>("/localization/ndt_status",10);
    prediction_sub_=create_subscription<nav_msgs::msg::Odometry>("/localization/ekf",100,[this](nav_msgs::msg::Odometry::ConstSharedPtr m){auto&p=m->pose.pose;Eigen::Quaternionf q(p.orientation.w,p.orientation.x,p.orientation.y,p.orientation.z);prediction_.setIdentity();prediction_.block<3,3>(0,0)=q.normalized().toRotationMatrix();prediction_(0,3)=p.position.x;prediction_(1,3)=p.position.y;prediction_(2,3)=p.position.z;prediction_time_=rclcpp::Time(m->header.stamp);});
    scan_sub_=create_subscription<sensor_msgs::msg::PointCloud2>("/localization/input_scan",rclcpp::QoS(2).reliable(),std::bind(&Localizer::scan,this,std::placeholders::_1));
    map_timer_=create_wall_timer(std::chrono::seconds(2),[this](){sensor_msgs::msg::PointCloud2 m;pcl::toROSMsg(*map_,m);m.header.frame_id="map";m.header.stamp=now();map_pub_->publish(m);});
    RCLCPP_INFO(get_logger(),"NDT map ready: %zu points; fixed map, no map updates",map_->size());
  }
  void scan(sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
    auto start=std::chrono::steady_clock::now();double t=rclcpp::Time(message->header.stamp).seconds();Eigen::Matrix4f guess=last_;
    if(initialized_ && prediction_time_.seconds()>=last_time_-.02 && std::abs(t-prediction_time_.seconds())<.5)guess=prediction_;
    if(have_velocity_ && t>last_time_ && t-last_time_<1.5)
      guess.block<3,1>(0,3)=last_.block<3,1>(0,3)+velocity_*float(t-last_time_);
    Eigen::Vector3f center=guess.block<3,1>(0,3);
    if((center-target_center_).norm()>10.){
      target_->clear();target_center_=center;
      for(auto&p:*map_)if((p.getVector3fMap()-center).squaredNorm()<90*90)target_->push_back(p);
      ndt_.setInputTarget(target_);target_tree_.setInputCloud(target_);
    }
    Cloud::Ptr input(new Cloud), source(new Cloud);pcl::fromROSMsg(*message,*input);pcl::VoxelGrid<pcl::PointXYZ> voxel;voxel.setInputCloud(input);voxel.setLeafSize(.65,.65,.65);voxel.filter(*source);
    bool accepted=false;double fitness=1e6,inlier=0.,jump=1e6,angle=1e6;Eigen::Matrix4f result=guess;Cloud aligned;
    int iterations=0;
    if(source->size()>100 && target_->size()>1000){ndt_.setInputSource(source);ndt_.align(aligned,guess);result=ndt_.getFinalTransformation();fitness=ndt_.getFitnessScore(2.);jump=(result.block<3,1>(0,3)-guess.block<3,1>(0,3)).norm();angle=Eigen::AngleAxisf(guess.block<3,3>(0,0).transpose()*result.block<3,3>(0,0)).angle();iterations=ndt_.getFinalNumIteration();
      std::vector<int> ids(1);std::vector<float> distances(1);int hit=0,count=0;
      for(size_t k=0;k<aligned.size();k+=3){++count;if(target_tree_.nearestKSearch(aligned[k],1,ids,distances)&&distances[0]<.64)++hit;}inlier=double(hit)/std::max(count,1);
      accepted=ndt_.hasConverged()&&iterations<max_iterations_&&result.allFinite()&&fitness<.3&&inlier>.75&&jump<2.&&angle<.2;
    }
    if(accepted){if(initialized_ && t>last_time_ && t-last_time_<1.5){velocity_=(result.block<3,1>(0,3)-last_.block<3,1>(0,3))/float(t-last_time_);if(velocity_.norm()<25.)have_velocity_=true;else have_velocity_=false;}last_=result;last_time_=t;initialized_=true;nav_msgs::msg::Odometry odom;odom.header=message->header;odom.header.frame_id="map";odom.child_frame_id="localization_base";odom.pose.pose.position.x=result(0,3);odom.pose.pose.position.y=result(1,3);odom.pose.pose.position.z=result(2,3);Eigen::Quaternionf q(result.block<3,3>(0,0));odom.pose.pose.orientation.x=q.x();odom.pose.pose.orientation.y=q.y();odom.pose.pose.orientation.z=q.z();odom.pose.pose.orientation.w=q.w();double variance=std::max(.01,std::min(.25,fitness*.2));for(int k=0;k<3;++k)odom.pose.covariance[k*7]=variance;for(int k=3;k<6;++k)odom.pose.covariance[k*7]=.0012;pose_pub_->publish(odom);
      sensor_msgs::msg::PointCloud2 scan;pcl::toROSMsg(aligned,scan);scan.header=odom.header;scan_pub_->publish(scan);
    }
    Eigen::Quaternionf q(result.block<3,3>(0,0));
    double duration=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::ostringstream ss;ss.precision(17);ss<<"{\"stamp_s\":"<<t<<",\"accepted\":"<<(accepted?"true":"false")<<",\"iterations\":"<<iterations<<",\"pose\":["<<result(0,3)<<","<<result(1,3)<<","<<result(2,3)<<","<<q.x()<<","<<q.y()<<","<<q.z()<<","<<q.w()<<"],\"fitness_m2\":"<<fitness<<",\"inlier_fraction\":"<<inlier<<",\"correction_m\":"<<jump<<",\"correction_rad\":"<<angle<<",\"compute_s\":"<<duration<<"}";std_msgs::msg::String status;status.data=ss.str();status_pub_->publish(status);RCLCPP_INFO(get_logger(),"%s",status.data.c_str());
  }
};
int main(int argc,char**argv){rclcpp::init(argc,argv);try{rclcpp::spin(std::make_shared<Localizer>());}catch(std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}rclcpp::shutdown();}
