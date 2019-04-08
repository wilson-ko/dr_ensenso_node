cmake_minimum_required(VERSION 3.8)
project(dr_ensenso_node)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra -Wpedantic)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(catkin REQUIRED COMPONENTS
	cv_bridge
	dr_eigen
	dr_ensenso
	dr_ensenso_msgs
	dr_msgs
	dr_opencv
	dr_pcl
	dr_ros
	dr_thread
	dr_util
	estd
	image_transport
	pcl_ros
	roscpp
	tf2
	tf2_ros
)

find_package(systemd)

catkin_package(
	INCLUDE_DIRS include
	LIBRARIES ensenso_calibrator
	CATKIN_DEPENDS dr_eigen
)

include_directories(SYSTEM ${catkin_INCLUDE_DIRS})
include_directories(include/${PROJECT_NAME})

add_executable(ensenso            src/ensenso.cpp)
add_executable(fake_ensenso       src/fake_ensenso.cpp)
add_executable(calibrate          src/calibrate.cpp)
add_library   (ensenso_calibrator src/ensenso_calibrator.cpp)
target_link_libraries(ensenso            ${catkin_LIBRARIES} systemd)
target_link_libraries(fake_ensenso       ${catkin_LIBRARIES})
target_link_libraries(calibrate          ${catkin_LIBRARIES})
target_link_libraries(ensenso_calibrator ${catkin_LIBRARIES})

install(
	TARGETS ensenso fake_ensenso calibrate ensenso_calibrator
	ARCHIVE DESTINATION "${CATKIN_PACKAGE_LIB_DESTINATION}"
	LIBRARY DESTINATION "${CATKIN_PACKAGE_LIB_DESTINATION}"
	RUNTIME DESTINATION "${CATKIN_PACKAGE_BIN_DESTINATION}"
)
