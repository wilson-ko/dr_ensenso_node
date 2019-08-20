#include <ros/ros.h>
#include <dr_ensenso_msgs/Calibrate.h>
#include <dr_ensenso_msgs/FinalizeCalibration.h>
#include <dr_ensenso_msgs/InitializeCalibration.h>
#include <dr_ensenso_msgs/GetCameraData.h>
#include <dr_ensenso_msgs/DetectCalibrationPattern.h>
#include <dr_msgs/SendPose.h>
#include <dr_msgs/SendPoseStamped.h>

#include <dr_param/param.hpp>
#include <dr_ros/node.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/image_encodings.h>
#include <std_srvs/Empty.h>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>

#include <boost/filesystem.hpp>

namespace dr {

class FakeEnsensoNode : public Node {
private:
	/// Service server for supplying point clouds and images.
	ros::ServiceServer get_data_server; //NOLINT

	/// Buffered image.
	cv::Mat image; //NOLINT

	/// Buffered point cloud.
	pcl::PointCloud<pcl::PointXYZ> point_cloud; //NOLINT

	/// Name for the camera frame.
	std::string camera_frame; //NOLINT

	/// If true, publishes point cloud data when calling getData.
	bool publish_cloud = true; //NOLINT

	/// If true, publishes point cloud data when calling getData.
	bool publish_image = true; //NOLINT

	/// Object for transporting images.
	image_transport::ImageTransport image_transport; //NOLINT

	struct {
		/// Service server for supplying point clouds and images.
		ros::ServiceServer camera_data; //NOLINT

		/// Service server for dumping image and cloud to disk.
		ros::ServiceServer dump_data; //NOLINT

		/// Service server for retrieving the pose of the pattern.
		ros::ServiceServer detect_calibration_pattern; //NOLINT

		/// Service server for initializing the calibration sequence.
		ros::ServiceServer initialize_calibration; //NOLINT

		/// Service server for recording one calibration sample.
		ros::ServiceServer record_calibration; //NOLINT

		/// Service server for finalizing the calibration.
		ros::ServiceServer finalize_calibration; //NOLINT

		/// Service server for setting the camera pose setting of the Ensenso.
		ros::ServiceServer set_workspace_calibration; //NOLINT

		/// Service server for clearing the camera pose setting of the Ensenso.
		ros::ServiceServer clear_workspace_calibration; //NOLINT

		/// Service server combining 'get_pattern_pose', 'set_workspace', and stores it to the ensenso.
		ros::ServiceServer calibrate_workspace; //NOLINT

		/// Service server for storing the calibration.
		ros::ServiceServer store_workspace_calibration; //NOLINT
	} servers; //NOLINT

	struct {
		ros::Publisher cloud; //NOLINT
		image_transport::Publisher image; //NOLINT
	} publishers; //NOLINT

public:
	FakeEnsensoNode() : image_transport(*this) {
		camera_frame                        = getParam<std::string>("camera_frame");
		publish_cloud                       = getParam<bool>("publish_cloud", publish_cloud, true);
		publish_image                       = getParam<bool>("publish_image", publish_image, true);

		publishers.cloud                    = advertise<sensor_msgs::PointCloud2>("cloud", 1, true);
		publishers.image                    = image_transport.advertise("image", 1, true);

		servers.camera_data                 = advertiseService("get_data"                    , &FakeEnsensoNode::onGetData                   , this);
		servers.dump_data                   = advertiseService("dump_data"                   , &FakeEnsensoNode::onDumpData                  , this);
		servers.detect_calibration_pattern  = advertiseService("detect_calibration_pattern"  , &FakeEnsensoNode::onDetectCalibrationPattern  , this);
		servers.initialize_calibration      = advertiseService("initialize_calibration"      , &FakeEnsensoNode::onInitializeCalibration     , this);
		servers.record_calibration          = advertiseService("record_calibration"          , &FakeEnsensoNode::onRecordCalibration         , this);
		servers.finalize_calibration        = advertiseService("finalize_calibration"        , &FakeEnsensoNode::onFinalizeCalibration       , this);
		servers.set_workspace_calibration   = advertiseService("set_workspace_calibration"   , &FakeEnsensoNode::onSetWorkspaceCalibration   , this);
		servers.clear_workspace_calibration = advertiseService("clear_workspace_calibration" , &FakeEnsensoNode::onClearWorkspaceCalibration , this);
		servers.calibrate_workspace         = advertiseService("calibrate_workspace"         , &FakeEnsensoNode::onCalibrateWorkspace        , this);
		servers.store_workspace_calibration = advertiseService("store_workspace_calibration" , &FakeEnsensoNode::onStoreWorkspaceCalibration , this);
		DR_SUCCESS("Fake Ensenso node initialized.");
	}

private:
	bool onGetData(dr_ensenso_msgs::GetCameraData::Request & /*unused_req*/, dr_ensenso_msgs::GetCameraData::Response & res) {
		DR_INFO("Received data request.");

		// read image file path
		std::string image_file = getParam<std::string>("image_path");
		std::string point_cloud_file = getParam<std::string>("point_cloud_path");

		if (!boost::filesystem::exists(image_file))       { DR_ERROR("Failed to load image: File does not exist: " << image_file); }
		if (!boost::filesystem::exists(point_cloud_file)) { DR_ERROR("Failed to load point cloud: File does not exist: " << point_cloud_file); }

		// load image
		image = cv::imread(image_file, cv::IMREAD_UNCHANGED);
		if (image.empty()) {
			DR_ERROR("Failed to load image from path: " << image_file);
			return false;
		}

		// load point cloud
		if (pcl::io::loadPCDFile(point_cloud_file, point_cloud) == -1 || point_cloud.empty()) {
			DR_ERROR("Failed to load point cloud from path: " << point_cloud_file);
			return false;
		}

		// Prepare the header.
		std_msgs::Header header;
		header.frame_id = camera_frame;
		header.stamp = ros::Time::now();

		// Prepare image conversion.
		cv_bridge::CvImage cv_image(
			header,
			"bgr8", // This might be incorrect?
			//std::move(image) No move will actually happen since image is const ref.
			image
		);

		// Copy the image.
		res.color = *cv_image.toImageMsg();

		// copy the point cloud
		pcl::toROSMsg(point_cloud, res.point_cloud);
		res.point_cloud.header = header;

		// publish point cloud if requested
		if (publish_cloud) {
			DR_SUCCESS("Publishing cloud");
			publishers.cloud.publish(res.point_cloud);
		}

		// publish image if requested
		if (publish_image) {
			DR_SUCCESS("Publishing image");
			publishers.image.publish(res.color);
		}

		return true;
	}

	bool onDumpData(std_srvs::Empty::Request & /*unused_req*/, std_srvs::Empty::Response & /*unused_res*/) {
		DR_ERROR("The dump_data service is not implemented in the fake ensenso node.");
		return false;
	}

	bool onDetectCalibrationPattern(dr_ensenso_msgs::DetectCalibrationPattern::Request & /*unused_req*/, dr_ensenso_msgs::DetectCalibrationPattern::Response & /*unused res*/) {
		DR_ERROR("The get_pattern_pose service is not implemented in the fake ensenso node.");
		return false;
	}

	bool onInitializeCalibration(dr_ensenso_msgs::InitializeCalibration::Request & /*unused_req*/, dr_ensenso_msgs::InitializeCalibration::Response & /*unused_res*/) {
		DR_ERROR("The initialize_calibration service is not implemented in the fake ensenso node.");
		return false;
	}

	bool onRecordCalibration(dr_msgs::SendPose::Request & /*unused_req*/, dr_msgs::SendPose::Response & /*unused_res*/) {
		DR_ERROR("The record_calibration service is not implemented in the fake ensenso node.");
		return false;
	}

	bool onFinalizeCalibration(dr_ensenso_msgs::FinalizeCalibration::Request & /*unused_req*/, dr_ensenso_msgs::FinalizeCalibration::Response & /*unused_res*/) {
		DR_ERROR("The finalize_calibration service is not implemented in the fake ensenso node.");
		return false;
	}

	bool onSetWorkspaceCalibration(dr_msgs::SendPoseStamped::Request & /*unused_req*/, dr_msgs::SendPoseStamped::Response & /*unused_res*/) {
		DR_ERROR("The set_workspace service is not implemented in the fake ensenso node.");
		return false;
	}

	bool onClearWorkspaceCalibration(std_srvs::Empty::Request & /*unused_req*/, std_srvs::Empty::Response & /*unuses_res*/) {
		DR_ERROR("The clear_workspace service is not implemented in the fake ensenso node.");
		return false;
	}

	bool onCalibrateWorkspace(dr_ensenso_msgs::Calibrate::Request & /*unused_req*/, dr_ensenso_msgs::Calibrate::Response & /*unused_res*/) {
		DR_ERROR("The calibrate service is not implemented in the fake ensenso node.");
		return false;
	}

	bool onStoreWorkspaceCalibration(std_srvs::Empty::Request & /*unused_req*/, std_srvs::Empty::Response & /*unused_res*/) {
		DR_ERROR("The store_calibration service is not implemented in the fake ensenso node.");
		return false;
	}

};

} //namespace dr

int main(int argc, char ** argv) {
	ros::init(argc, argv, "ensenso");
	dr::FakeEnsensoNode node;
	ros::spin();
}
