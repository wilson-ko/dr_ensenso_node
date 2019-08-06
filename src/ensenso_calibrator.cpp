#include "ensenso_calibrator.hpp"

namespace dr {

EnsensoCalibratorNode::EnsensoCalibratorNode(
	std::string const & initialize_calibration_service,
	std::string const & record_calibration_service,
	std::string const & finalize_calibration_service,
	bool wait_for_services,
	bool store_calibration
) : store_calibration_{store_calibration} {
	// Connect the services to dr_ensenso_node.
	// TODO(wko): Ignoring clang-tidy now, but this seems pretty serious.
	services_.initialize_calibration.connect(*this, initialize_calibration_service, wait_for_services); //NOLINT
	services_.record_calibration    .connect(*this, record_calibration_service, wait_for_services);     //NOLINT
	services_.finalize_calibration  .connect(*this, finalize_calibration_service, wait_for_services);   //NOLINT
}

estd::result<void, estd::error> EnsensoCalibratorNode::initializeCalibration(InitializeCalibrationConfig const & config) {
	// Copy the configuration to the request.
	dr_ensenso_msgs::InitializeCalibrationRequest request;
	// ROS stores bool types from srv/msg files as uint8_t..?
	request.camera_moving = config.camera_moving; //NOLINT
	request.fixed_frame   = config.fixed_frame;
	request.moving_frame  = config.moving_frame;
	request.dump_dir      = config.dump_dir;

	if (config.camera_guess)  { request.camera_guess  = dr::toRosPose(*config.camera_guess); }
	if (config.pattern_guess) { request.pattern_guess = dr::toRosPose(*config.pattern_guess); }

	// Try to call the service.
	try {
		services_.initialize_calibration(request);
	} catch (dr::ServiceError const & e) {
		return estd::error{e.what()};
	}

	return estd::in_place_valid;
}

estd::result<void, estd::error> EnsensoCalibratorNode::recordCalibration(Eigen::Isometry3d const & robot_pose) {
	// Construct the request.
	dr_msgs::SendPoseRequest request;
	request.data = dr::toRosPose(robot_pose);

	// Try to call the service.
	try {
		services_.record_calibration(request);
	} catch (dr::ServiceError const & e) {
		return estd::error{e.what()};
	}

	return estd::in_place_valid;
}

estd::result<EnsensoCalibratorNode::CalibrationResult, estd::error> EnsensoCalibratorNode::finalizeCalibration() {
	// Construct the request.
	dr_ensenso_msgs::FinalizeCalibrationRequest request;
	request.store_calibration = store_calibration_; //NOLINT

	// Try to call the service.
	try {
		dr_ensenso_msgs::FinalizeCalibrationResponse response = services_.finalize_calibration(request);

		return CalibrationResult{
			dr::toEigen(response.camera_pose.pose),
			dr::toEigen(response.pattern_pose.pose),
			response.residual_error
		};
	} catch (dr::ServiceError const & e) {
		return estd::error{e.what()};
	}
}

} //namespace dr
