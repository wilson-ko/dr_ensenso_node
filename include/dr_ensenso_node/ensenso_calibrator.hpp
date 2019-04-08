#pragma once

#include <dr_ensenso_msgs/InitializeCalibration.h>
#include <dr_msgs/SendPose.h>
#include <dr_ensenso_msgs/FinalizeCalibration.h>
#include <dr_ros/service_client.hpp>
#include <dr_ros/node.hpp>
#include <dr_eigen/ros.hpp>

#include <estd/result.hpp>

#include <ros/node_handle.h>

#include <Eigen/Eigen>

namespace dr {

class EnsensoCalibratorNode: public Node {
public:
	struct InitializeCalibrationConfig {
		/// If true, the camera is attached to the moving frame; if false, it is attached to the fixed frame.
		/// Additionally, the calibration plate is assumed to be attached to the other frame.
		bool camera_moving;

		/// Name of the moving frame (usually the name of the robot end effector).
		std::string moving_frame;

		/// Name of the fixed frame (usually the name of the robot base frame).
		std::string fixed_frame;

		/// Initial guess of the pose of the camera, can be used to speed up the calibration optimization.
		/// Depending on the value of camera_moving, this pose is defined in either moving_frame or fixed_frame.
		Eigen::Isometry3d camera_guess;

		/// Initial guess of the pattern, can be used to speed up the calibration optimization.
		/// Depending on the value of camera_moving, this pose is defined in either moving_frame or fixed_frame.
		Eigen::Isometry3d pattern_guess;
	};

	struct CalibrationResult {
		/// Calibrated pose of the camera in either moving_frame or fixed_frame (depending on camera_moving).
		Eigen::Isometry3d camera_pose;

		/// Calibrated pose of the pattern in either moving_frame or fixed_frame (depending on camera_moving).
		Eigen::Isometry3d pattern_pose;

		/// The residual error of the calibration.
		double residual_error;
	};

private:
	/// List of ros services.
	struct Services {
		/// Service for initializing a calibration sequence.
		dr::ServiceClient<dr_ensenso_msgs::InitializeCalibration> initialize_calibration;

		/// Service for recording a calibration pattern.
		dr::ServiceClient<dr_msgs::SendPose> record_calibration;

		/// Service for finalizing a calibration sequence.
		dr::ServiceClient<dr_ensenso_msgs::FinalizeCalibration> finalize_calibration;
	} services_;

public:
	EnsensoCalibratorNode(
		std::string const & initialize_calibration_service, ///< Service name for the initialization of the calibration.
		std::string const & record_calibration_service,     ///< Service name for the detect calibration pattern service.
		std::string const & finalize_calibration_service,   ///< Service name for finalizing the calibration.
		bool wait_for_services = false                      ///< If true, waits for the services to come alive.
	);

	/// Initializes a calibration sequence, clearing any state from previous calibration sequences.
	estd::result<void, estd::error> initializeCalibration(InitializeCalibrationConfig const & config);

	/// Records a single calibration pattern.
	estd::result<void, estd::error> recordCalibration(Eigen::Isometry3d const & robot_pose);

	/// Finalizes the calibration, returning the result.
	estd::result<CalibrationResult, estd::error> finalizeCalibration();
};

}
