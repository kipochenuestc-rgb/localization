cmake_minimum_required(VERSION 3.16)
project(ndt_localization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io kdtree registration)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rmw REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(localization_geometry localization_geometry.cpp)
target_include_directories(localization_geometry PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(localization_geometry Eigen3::Eigen ${PCL_LIBRARIES} yaml-cpp)

add_executable(localization_frontend localization_frontend.cpp)
target_include_directories(localization_frontend PRIVATE ${PCL_INCLUDE_DIRS})
ament_target_dependencies(localization_frontend
  geometry_msgs message_filters nav_msgs pcl_conversions rclcpp rmw sensor_msgs)
target_link_libraries(localization_frontend localization_geometry ${PCL_LIBRARIES})

add_executable(ndt_localizer ndt_localizer.cpp)
target_include_directories(ndt_localizer PRIVATE ${PCL_INCLUDE_DIRS})
ament_target_dependencies(ndt_localizer nav_msgs pcl_conversions rclcpp sensor_msgs std_msgs)
target_link_libraries(ndt_localizer localization_geometry ${PCL_LIBRARIES})

install(TARGETS localization_frontend ndt_localizer
  DESTINATION lib/${PROJECT_NAME})
ament_package()
