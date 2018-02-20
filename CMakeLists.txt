cmake_minimum_required(VERSION 3.8)
project(dr_ensenso_node)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
	cv_bridge
	dr_eigen
	dr_ensenso
	dr_ensenso_msgs
	dr_msgs
	dr_opencv
	dr_ros
	dr_thread
	dr_util
	image_transport
	pcl_conversions
	pcl_ros
	roscpp
	tf2
	tf2_ros
)

catkin_package()

include_directories(SYSTEM ${catkin_INCLUDE_DIRS})

add_executable(ensenso      src/ensenso.cpp)
add_executable(fake_ensenso src/fake_ensenso.cpp)
add_executable(calibrate    src/calibrate.cpp)
target_link_libraries(ensenso      ${catkin_LIBRARIES})
target_link_libraries(fake_ensenso ${catkin_LIBRARIES})
target_link_libraries(calibrate    ${catkin_LIBRARIES})

install(
	TARGETS ensenso fake_ensenso calibrate
	ARCHIVE DESTINATION "${CATKIN_PACKAGE_LIB_DESTINATION}"
	LIBRARY DESTINATION "${CATKIN_PACKAGE_LIB_DESTINATION}"
	RUNTIME DESTINATION "${CATKIN_PACKAGE_BIN_DESTINATION}"
)
