#include <dr_eigen/ros.hpp>
#include <dr_eigen/yaml.hpp>
#include <dr_ensenso/ensenso.hpp>
#include <dr_ensenso/util.hpp>
#include <dr_ros/node.hpp>
#include <dr_thread/thread_pool.hpp>
#include <dr_util/timestamp.hpp>

#include <dr_ensenso_msgs/Calibrate.h>
#include <dr_ensenso_msgs/FinalizeCalibration.h>
#include <dr_ensenso_msgs/GetCameraData.h>
#include <dr_ensenso_msgs/DetectCalibrationPattern.h>
#include <dr_ensenso_msgs/InitializeCalibration.h>
#include <dr_msgs/SendPose.h>
#include <dr_msgs/SendPoseStamped.h>

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <std_srvs/Empty.h>

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <boost/optional.hpp>

#include <memory>

namespace dr {

ImageType parseImageType(std::string const & name) {
	if (name == "stereo_raw_left"       ) return ImageType::stereo_raw_left;
	if (name == "stereo_raw_right"      ) return ImageType::stereo_raw_right;
	if (name == "stereo_rectified_left" ) return ImageType::stereo_rectified_left;
	if (name == "stereo_rectified_right") return ImageType::stereo_rectified_right;
	if (name == "disparity"             ) return ImageType::disparity;
	if (name == "monocular_raw"         ) return ImageType::monocular_raw;
	if (name == "monocular_rectified"   ) return ImageType::monocular_rectified;
	throw std::runtime_error("Unknown image type: " + name);
}

std::string encoding(ImageType type) {
	switch (type) {
		case ImageType::stereo_raw_left:
		case ImageType::stereo_raw_right:
		case ImageType::stereo_rectified_left:
		case ImageType::stereo_rectified_right:
			return sensor_msgs::image_encodings::MONO8;
		case ImageType::disparity:
			return sensor_msgs::image_encodings::MONO16;
		case ImageType::monocular_raw:
		case ImageType::monocular_rectified:
		case ImageType::monocular_overlay:
			return sensor_msgs::image_encodings::BGR8;
	}

	throw std::runtime_error("Unknown image type: " + std::to_string(int(type)));
}

bool isMonocular(ImageType type) {
	switch (type) {
		case ImageType::stereo_raw_left:
		case ImageType::stereo_raw_right:
		case ImageType::stereo_rectified_left:
		case ImageType::stereo_rectified_right:
		case ImageType::disparity:
			return false;
		case ImageType::monocular_raw:
		case ImageType::monocular_rectified:
		case ImageType::monocular_overlay:
			return true;
	}

	throw std::runtime_error("Unknown image type: " + std::to_string(int(type)));
}

bool needsRectification(ImageType type) {
	switch (type) {
		case ImageType::stereo_rectified_left:
		case ImageType::stereo_rectified_right:
		case ImageType::disparity:
		case ImageType::monocular_rectified:
		case ImageType::monocular_overlay:
			return true;
		case ImageType::monocular_raw:
		case ImageType::stereo_raw_left:
		case ImageType::stereo_raw_right:
			return false;
	}
	throw std::runtime_error("Unknown image type: " + std::to_string(int(type)));
}

class EnsensoNode: public Node {
	/// The wrapper for the Ensenso stereo camera.
	std::unique_ptr<dr::Ensenso> ensenso_camera;

	/// Serial id of the Ensenso camera.
	std::string serial;

	/// The type of the color image.
	ImageType image_source;

	/// If true, registers the point clouds.
	bool register_pointcloud;

	/// If true, retrieves the monocular camera and Ensenso simultaneously. A hardware trigger is advised to remove the projector from the uEye image.
	bool synced_retrieve;

	/// The frame in which the image and point clouds are send.
	std::string camera_frame;

	struct {
		/// Frame to calibrate the camera to when camera_moving is true (gripper frame).
		std::string moving_frame;

		/// Frame to calibrate the camera to when camera_moving is false (robot origin or world frame).
		std::string fixed_frame;

		/// Used in calibration. Determines if the camera is moving (eye in hand) or static.
		bool camera_moving;

		// Guess of the camera pose relative to gripper (for moving camera) or relative to robot origin (for static camera).
		boost::optional<Eigen::Isometry3d> camera_guess;

		// Guess of the calibration pattern pose relative to gripper (for static camera) or relative to robot origin (for moving camera).
		boost::optional<Eigen::Isometry3d> pattern_guess;

		/// List of robot poses corresponding to the list of recorded calibration patterns.
		std::vector<Eigen::Isometry3d> robot_poses;
	} auto_calibration;

	/// If true, publishes recorded data.
	bool publish_data;

	/// If true, save recorded data to disk.
	bool save_data;

	/// Location where the images and point clouds are stored.
	std::string camera_data_path;

	/// Thread pool for parallel work.
	dr::ThreadPool thread_pool;

	struct {
		/// Service server for supplying point clouds and images.
		ros::ServiceServer camera_data;

		/// Service server for dumping image and cloud to disk.
		ros::ServiceServer save_data;

		/// Service server for retrieving the pose of the pattern.
		ros::ServiceServer get_pattern_pose;

		/// Service server for setting the camera pose setting of the Ensenso.
		ros::ServiceServer set_workspace_calibration;

		/// Service server for clearing the camera pose setting of the Ensenso.
		ros::ServiceServer clear_workspace_calibration;

		/// Service server combining 'get_pattern_pose', 'set_workspace', and stores it to the ensenso.
		ros::ServiceServer calibrate_workspace;

		/// Service server for storing the calibration.
		ros::ServiceServer store_workspace_calibration;

		/// Service server for initializing the calibration sequence.
		ros::ServiceServer initialize_calibration;

		/// Service server for recording one calibration sample.
		ros::ServiceServer record_calibration;

		/// Service server for finalizing the calibration.
		ros::ServiceServer finalize_calibration;
	} servers;

	struct Publishers {
		/// Publisher for the calibration result.
		ros::Publisher calibration;

		/// Publisher for publishing raw point clouds.
		ros::Publisher cloud;

		/// Publisher for publishing images.
		image_transport::Publisher image;

		/// Publisher for publishing live images.
		image_transport::Publisher live;
	} publishers;

	/// Object for handling transportation of images.
	image_transport::ImageTransport image_transport;

	/// Timer to trigger calibration publishing.
	ros::Timer publish_calibration_timer;

	/// Timer to trigger image publishing.
	ros::Timer publish_images_timer;

public:
	EnsensoNode() : thread_pool(1), image_transport(*this) {
		configure();
	}

private:
	/// Resets calibration state from this node.
	void resetCalibration() {
		auto_calibration.moving_frame  = "";
		auto_calibration.fixed_frame   = "";
		auto_calibration.camera_guess  = boost::none;
		auto_calibration.pattern_guess = boost::none;
		auto_calibration.robot_poses.clear();
	}

protected:
	using Point = pcl::PointXYZ;
	using PointCloud = pcl::PointCloud<Point>;

	struct Data {
		PointCloud cloud;
		cv::Mat image;
	};

	void configure() {
		// load ROS parameters
		camera_frame        = getParam<std::string>("camera_frame", "camera_frame");
		camera_data_path    = getParam<std::string>("camera_data_path", "camera_data");
		image_source        = parseImageType(getParam<std::string>("image_source"));
		register_pointcloud = getParam<bool>("register_point_cloud");
		synced_retrieve     = getParam<bool>("synced_retrieve", true);
		publish_data        = getParam<bool>("publish_data",    true);
		save_data           = getParam<bool>("save_data",       true);

		// get Ensenso serial
		serial = getParam<std::string>("serial", "");
		if (serial != "") {
			DR_INFO("Opening Ensenso with serial '" << serial << "'...");
		} else {
			DR_INFO("Opening first available Ensenso...");
		}

		try {
			// create the camera
			ensenso_camera = std::make_unique<dr::Ensenso>(serial, needMonocular());
		} catch (dr::NxError const & e) {
			throw std::runtime_error("Failed initializing camera. " + std::string(e.what()));
		} catch (std::runtime_error const & e) {
			throw std::runtime_error("Failed initializing camera. " + std::string(e.what()));
		}

		// activate service servers
		servers.camera_data                 = advertiseService("get_data"                   , &EnsensoNode::onGetData                  , this);
		servers.save_data                   = advertiseService("save_data"                  , &EnsensoNode::onSaveData                 , this);
		servers.get_pattern_pose            = advertiseService("detect_calibration_pattern" , &EnsensoNode::onDetectCalibrationPattern , this);
		servers.initialize_calibration      = advertiseService("initialize_calibration"     , &EnsensoNode::onInitializeCalibration    , this);
		servers.record_calibration          = advertiseService("record_calibration"         , &EnsensoNode::onRecordCalibration        , this);
		servers.finalize_calibration        = advertiseService("finalize_calibration"       , &EnsensoNode::onFinalizeCalibration      , this);
		servers.set_workspace_calibration   = advertiseService("set_workspace_calibration"  , &EnsensoNode::onSetWorkspaceCalibration  , this);
		servers.clear_workspace_calibration = advertiseService("clear_workspace_calibration", &EnsensoNode::onClearWorkspaceCalibration, this);
		servers.calibrate_workspace         = advertiseService("calibrate_workspace"        , &EnsensoNode::onCalibrateWorkspace       , this);
		servers.store_workspace_calibration = advertiseService("store_workspace_calibration", &EnsensoNode::onStoreWorkspaceCalibration, this);

		// activate publishers
		publishers.calibration = advertise<geometry_msgs::PoseStamped>("calibration", 1, true);
		publishers.cloud       = advertise<PointCloud>("cloud", 1, true);
		publishers.image       = image_transport.advertise("image", 1, true);
		publishers.live        = image_transport.advertise("live", 1, true);

		// load ensenso parameters file
		std::string ensenso_param_file = getParam<std::string>("ensenso_param_file", "");
		if (ensenso_param_file != "") {
			try {
				if (!ensenso_camera->loadParameters(ensenso_param_file)) {
					DR_ERROR("Failed to set Ensenso params. File path: " << ensenso_param_file);
				}
			} catch (dr::NxError const & e) {
				DR_ERROR("Failed to set Ensenso params. " << e.what());
			}
		}

		if (ensenso_camera->hasMonocular()) {
			// load monocular parameters file
			std::string monocular_param_file = getParam<std::string>("monocular_param_file", "");
			if (monocular_param_file != "") {
				try {
					if (!ensenso_camera->loadMonocularParameters(monocular_param_file)) {
						DR_ERROR("Failed to set monocular camera params. File path: " << monocular_param_file);
					}
				} catch (dr::NxError const & e) {
					DR_ERROR("Failed to set monocular camera params. " << e.what());
				}
			}

			// load monocular parameter set file
			std::string monocular_ueye_param_file = getParam<std::string>("monocular_ueye_param_file", "");
			if (monocular_ueye_param_file != "") {
				try {
					ensenso_camera->loadMonocularUeyeParameters(monocular_ueye_param_file);
				} catch (dr::NxError const & e) {
					DR_ERROR("Failed to set monocular param set file. " << e.what());
				}
			}
		}

		// initialize other member variables
		resetCalibration();

		// start publish calibration timer
		double calibration_timer_rate = getParam("calibration_timer_rate", -1);
		if (calibration_timer_rate > 0) {
			publish_calibration_timer = createTimer(ros::Duration(calibration_timer_rate), &EnsensoNode::publishCalibration, this);
		}

		// start image publishing timer
		double publish_images_rate = getParam("publish_images_rate", 30);
		if (publish_images_rate > 0) {
			publish_images_timer = createTimer(ros::Rate(publish_images_rate), &EnsensoNode::publishImage, this);
		}

		DR_SUCCESS("Ensenso opened successfully.");
	}

	bool needMonocular() {
		return register_pointcloud || isMonocular(image_source);
	}

	void publishImage(ros::TimerEvent const &) {
		if (publishers.image.getNumSubscribers() == 0) return;

		captureData(false);

		// Create a header.
		std_msgs::Header header;
		header.frame_id = camera_frame;
		header.stamp    = ros::Time::now();

		// Prepare message.
		cv_bridge::CvImage cv_image(
			header,
			encoding(image_source),
			getImage()
		);

		// Publish the image to the live stream.
		publishers.live.publish(cv_image.toImageMsg());
	}


	void saveData(PointCloud const & point_cloud, cv::Mat const & image) {
		// create path if it does not exist
		boost::filesystem::path path(camera_data_path);
		if (!boost::filesystem::is_directory(path)) {
			boost::filesystem::create_directory(camera_data_path);
		}

		std::string time_string = getTimeString();

		pcl::io::savePCDFileBinary(camera_data_path + "/" + time_string + ".pcd", point_cloud);
		cv::imwrite(camera_data_path + "/" + time_string + ".png", image);
	}

	void captureData(bool point_cloud = true) {
		if (synced_retrieve) {
			ensenso_camera->retrieve(true, 1500, true, needMonocular());
		} else {
			if (needMonocular()) ensenso_camera->retrieve(true, 1500, false, true);
			ensenso_camera->retrieve(true, 1500, true, false);
		}
		if (point_cloud || needsRectification(image_source))     ensenso_camera->rectifyImages();
		if (point_cloud || image_source == ImageType::disparity) ensenso_camera->computeDisparity();
		if (point_cloud)                                       ensenso_camera->computePointCloud();
		if (point_cloud && register_pointcloud)                ensenso_camera->registerPointCloud();
	}

	cv::Mat getImage() {
		return ensenso_camera->loadImage(image_source);
	}

	pcl::PointCloud<pcl::PointXYZ> getPointCloud() {
		if (register_pointcloud) {
			return ensenso_camera->loadRegisteredPointCloud();
		} else {
			return ensenso_camera->loadPointCloud();
		}
	}

	boost::optional<Data> getData() {
		captureData();
		return Data{getPointCloud(), getImage()};
	}

	bool onGetData(dr_ensenso_msgs::GetCameraData::Request &, dr_ensenso_msgs::GetCameraData::Response & res) {
		boost::optional<Data> data = getData();
		if (!data) return false;
		pcl::toROSMsg(data->cloud, res.point_cloud);

		// Get the image.
		cv_bridge::CvImage cv_image(
			res.point_cloud.header,
			encoding(image_source),
			data->image
		);
		res.color = *cv_image.toImageMsg();

		// Store image and point cloud.
		if (save_data) {
			thread_pool.enqueue(
				&EnsensoNode::saveData,
				this,
				data->cloud,
				data->image
			);
		}

		// publish point cloud if requested
		if (publish_data) {
			publishers.cloud.publish(data->cloud);
			publishers.image.publish(res.color);
		}

		return true;
	}

	bool onSaveData(std_srvs::Empty::Request &, std_srvs::Empty::Response &) {
		boost::optional<Data> data = getData();
		if (!data) return false;

			// store point cloud and image in a separate thread
			thread_pool.enqueue(
				&EnsensoNode::saveData,
				this,
				data->cloud,
				data->image
			);
		return true;
	}

	bool onDetectCalibrationPattern(dr_ensenso_msgs::DetectCalibrationPattern::Request & req, dr_ensenso_msgs::DetectCalibrationPattern::Response & res) {
		if (req.samples == 0) {
			DR_ERROR("Unable to get pattern pose. Number of samples is set to 0.");
			return false;
		}

		try {
			res.data = dr::toRosPose(ensenso_camera->detectCalibrationPattern(req.samples));
		} catch (dr::NxError const & e) {
			DR_ERROR("Failed to find calibration pattern. " << e.what());
			return false;
		}

		return true;
	}

	bool onInitializeCalibration(dr_ensenso_msgs::InitializeCalibration::Request & req, dr_ensenso_msgs::InitializeCalibration::Response &) {
		try {
			ensenso_camera->discardCalibrationPatterns();
			ensenso_camera->clearWorkspaceCalibration();
		} catch (dr::NxError const & e) {
			DR_ERROR("Failed to discard patterns. " << e.what());
			return false;
		}
		resetCalibration();

		auto_calibration.camera_moving = req.camera_moving;
		auto_calibration.moving_frame  = req.moving_frame;
		auto_calibration.fixed_frame   = req.fixed_frame;

		// check for valid camera guess
		if (req.camera_guess.position.x == 0 && req.camera_guess.position.y == 0 && req.camera_guess.position.z == 0 &&
			req.camera_guess.orientation.x == 0 && req.camera_guess.orientation.y == 0 && req.camera_guess.orientation.z == 0 && req.camera_guess.orientation.w == 0) {
			auto_calibration.camera_guess = boost::none;
		} else {
			auto_calibration.camera_guess = dr::toEigen(req.camera_guess);
		}

		// check for valid pattern guess
		if (req.pattern_guess.position.x == 0 && req.pattern_guess.position.y == 0 && req.pattern_guess.position.z == 0 &&
			req.pattern_guess.orientation.x == 0 && req.pattern_guess.orientation.y == 0 && req.pattern_guess.orientation.z == 0 && req.pattern_guess.orientation.w == 0) {
			auto_calibration.pattern_guess = boost::none;
		} else {
			auto_calibration.pattern_guess = dr::toEigen(req.pattern_guess);
		}

		// check for proper initialization
		if (auto_calibration.moving_frame == "" || auto_calibration.fixed_frame == "") {
			DR_ERROR("No calibration frame provided.");
			return false;
		}

		DR_INFO("Successfully initialized calibration sequence.");

		return true;
	}

	bool onRecordCalibration(dr_msgs::SendPose::Request & req, dr_msgs::SendPose::Response &) {
		// check for proper initialization
		if (auto_calibration.moving_frame == "" || auto_calibration.fixed_frame == "") {
			DR_ERROR("No calibration frame provided.");
			return false;
		}

		try {
			// record a pattern
			ensenso_camera->recordCalibrationPattern();

			// add robot pose to list of poses
			auto_calibration.robot_poses.push_back(dr::toEigen(req.data));
		} catch (dr::NxError const & e) {
			DR_ERROR("Failed to record calibration pattern. " << e.what());
			return false;
		}

		DR_INFO("Successfully recorded a calibration sample.");
		return true;
	}

	bool onFinalizeCalibration(dr_ensenso_msgs::FinalizeCalibration::Request &, dr_ensenso_msgs::FinalizeCalibration::Response & res) {
		auto const & camera_moving = auto_calibration.camera_moving;
		auto const & moving_frame  = auto_calibration.moving_frame;
		auto const & fixed_frame   = auto_calibration.fixed_frame;
		auto const & camera_guess  = auto_calibration.camera_guess;
		auto const & pattern_guess = auto_calibration.pattern_guess;
		auto const & robot_poses   = auto_calibration.robot_poses;

		// check for proper initialization
		if (moving_frame == "" || fixed_frame == "") {
			DR_ERROR("No calibration frame provided.");
			return false;
		}

		try {
			// perform calibration
			dr::Ensenso::CalibrationResult calibration =
				ensenso_camera->computeCalibration(
					robot_poses,
					camera_moving,
					camera_guess,
					pattern_guess,
					camera_moving ? moving_frame : fixed_frame
				);

			// copy result
			res.camera_pose        = dr::toRosPoseStamped(std::get<0>(calibration), camera_moving ? moving_frame :  fixed_frame);
			res.pattern_pose       = dr::toRosPoseStamped(std::get<1>(calibration), camera_moving ?  fixed_frame : moving_frame);
			res.iterations         = std::get<2>(calibration);
			res.reprojection_error = std::get<3>(calibration);

			// store result in camera
			ensenso_camera->storeWorkspaceCalibration();

			// clear state
			resetCalibration();
		} catch (dr::NxError const & e) {
			// clear state (?)
			resetCalibration();

			DR_ERROR("Failed to finalize calibration. " << e.what());
			return false;
		}

		DR_INFO("Successfully finished calibration sequence.");
		return true;
	}

	bool onSetWorkspaceCalibration(dr_msgs::SendPoseStamped::Request & req, dr_msgs::SendPoseStamped::Response &) {
		try {
			ensenso_camera->setWorkspaceCalibration(dr::toEigen(req.data.pose), req.data.header.frame_id, Eigen::Isometry3d::Identity());
		} catch (dr::NxError const & e) {
			DR_ERROR("Failed to set workspace calibration: " << e.what());
			return false;
		}
		return true;
	}

	bool onClearWorkspaceCalibration(std_srvs::Empty::Request &, std_srvs::Empty::Response &) {
		try {
			ensenso_camera->clearWorkspaceCalibration();
		} catch (dr::NxError const & e) {
			DR_ERROR("Failed to clear workspace calibration: " << e.what());
			return false;
		}
		return true;
	}

	bool onCalibrateWorkspace(dr_ensenso_msgs::Calibrate::Request & req, dr_ensenso_msgs::Calibrate::Response &) {
		DR_INFO("Performing workspace calibration.");

		if (req.frame_id.empty()) {
			DR_ERROR("Calibration frame not set. Can not calibrate.");
			return false;
		}

		try {
			Eigen::Isometry3d pattern_pose = ensenso_camera->detectCalibrationPattern(req.samples);
			DR_INFO("Found calibration pattern at:\n" << dr::toYaml(pattern_pose));
			DR_INFO("Defined pattern pose:\n" << dr::toYaml(dr::toEigen(req.pattern)));
			ensenso_camera->setWorkspaceCalibration(pattern_pose, req.frame_id, dr::toEigen(req.pattern), true);
		} catch (dr::NxError const & e) {
			DR_ERROR("Failed to calibrate camera pose. " << e.what());
			return false;
		}
		return true;
	}

	bool onStoreWorkspaceCalibration(std_srvs::Empty::Request &, std_srvs::Empty::Response &) {
		try {
			ensenso_camera->storeWorkspaceCalibration();
		} catch (dr::NxError const & e) {
			DR_ERROR("Failed to store calibration. " << e.what());
			return false;
		}
		return true;
	}

	void publishCalibration(ros::TimerEvent const &) {
		geometry_msgs::PoseStamped pose;
		std::string frame = ensenso_camera->getWorkspaceCalibrationFrame();
		if (!frame.empty()) {
			pose = dr::toRosPoseStamped(ensenso_camera->getWorkspaceCalibration()->inverse(), frame, ros::Time::now());
		} else {
			pose = dr::toRosPoseStamped(Eigen::Isometry3d::Identity(), "", ros::Time::now());
		}

		publishers.calibration.publish(pose);
	}
};

}

int main(int argc, char ** argv) {
	ros::init(argc, argv, "ensenso");
	dr::EnsensoNode node;
	ros::spin();
	DR_INFO("Closing camera. This may take a (long) while.");
}

