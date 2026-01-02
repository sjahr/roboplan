#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>

#include <roboplan/core/scene.hpp>
#include <roboplan_example_models/resources.hpp>
#include <roboplan_oink/tasks/frame.hpp>

namespace roboplan {

class FrameTaskTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Use UR5 robot for testing
    const auto model_prefix = example_models::get_package_models_dir();
    urdf_path_ = model_prefix / "ur_robot_model" / "ur5_gripper.urdf";
    srdf_path_ = model_prefix / "ur_robot_model" / "ur5_gripper.srdf";
    package_paths_ = {example_models::get_package_share_dir()};
    yaml_config_path_ = model_prefix / "ur_robot_model" / "ur5_config.yaml";

    scene_ = std::make_shared<Scene>("test_scene", urdf_path_, srdf_path_, package_paths_,
                                     yaml_config_path_);

    const auto& model = scene_->getModel();
    num_variables_ = model.nv;

    // Set a non-zero initial configuration
    Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
    q[0] = 0.5;   // shoulder_pan_joint
    q[1] = -0.5;  // shoulder_lift_joint
    q[2] = 1.0;   // elbow_joint
    scene_->setJointPositions(q);
  }

  std::filesystem::path urdf_path_;
  std::filesystem::path srdf_path_;
  std::vector<std::filesystem::path> package_paths_;
  std::filesystem::path yaml_config_path_;
  std::shared_ptr<Scene> scene_;
  int num_variables_;
};

// Test frame task construction
TEST_F(FrameTaskTest, Construction) {
  CartesianConfiguration target_pose;
  target_pose.tform = pinocchio::SE3::Identity();

  // Test default construction
  FrameTask task1("tool0", target_pose);
  EXPECT_EQ(task1.frame_name, "tool0");
  EXPECT_EQ(task1.gain, 1.0);
  EXPECT_EQ(task1.lm_damping, 0.0);

  // Test construction with custom parameters
  FrameTaskParams params{
      .task_weight = 2.0,
      .position_cost = 1.5,
      .orientation_cost = 0.5,
      .task_gain = 0.8,
      .lm_damping = 0.01,
  };
  FrameTask task2("tool0", target_pose, params);
  EXPECT_EQ(task2.gain, 0.8);
  EXPECT_EQ(task2.lm_damping, 0.01);
}

// Test error computation at identity pose
TEST_F(FrameTaskTest, ErrorAtIdentity) {
  // Get current end-effector pose
  const auto& data = scene_->getData();
  const auto frame_id = scene_->getFrameId("tool0");
  ASSERT_TRUE(frame_id.has_value());

  pinocchio::SE3 current_pose = data.oMf[frame_id.value()];

  // Create task with current pose as target (zero error expected)
  CartesianConfiguration target_pose;
  target_pose.tform = current_pose;

  FrameTask task("tool0", target_pose);

  // Compute error
  Eigen::VectorXd error;
  auto result = task.computeError(*scene_, error);

  ASSERT_TRUE(result.has_value()) << "computeError failed: " << result.error();
  EXPECT_EQ(error.size(), 6);

  // Error should be close to zero
  EXPECT_NEAR(error.norm(), 0.0, 1e-10);
}

// Test error computation with translation offset
TEST_F(FrameTaskTest, ErrorWithTranslation) {
  // Get current end-effector pose
  const auto& data = scene_->getData();
  const auto frame_id = scene_->getFrameId("tool0");
  ASSERT_TRUE(frame_id.has_value());

  pinocchio::SE3 current_pose = data.oMf[frame_id.value()];

  // Create target pose with 10cm translation in x
  Eigen::Vector3d translation_offset(0.1, 0.0, 0.0);
  pinocchio::SE3 target_pose = current_pose;
  target_pose.translation() += translation_offset;

  CartesianConfiguration target_config;
  target_config.tform = target_pose;

  FrameTask task("tool0", target_config);

  // Compute error
  Eigen::VectorXd error;
  auto result = task.computeError(*scene_, error);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(error.size(), 6);

  // Position error should be approximately the translation offset
  Eigen::Vector3d position_error = error.head<3>();
  EXPECT_NEAR(position_error.norm(), translation_offset.norm(), 1e-6);

  // Orientation error should be zero
  Eigen::Vector3d orientation_error = error.tail<3>();
  EXPECT_NEAR(orientation_error.norm(), 0.0, 1e-10);
}

// Test error computation with rotation offset
TEST_F(FrameTaskTest, ErrorWithRotation) {
  // Get current end-effector pose
  const auto& data = scene_->getData();
  const auto frame_id = scene_->getFrameId("tool0");
  ASSERT_TRUE(frame_id.has_value());

  pinocchio::SE3 current_pose = data.oMf[frame_id.value()];

  // Create target pose with 90 degree rotation around z-axis
  Eigen::AngleAxisd rotation(M_PI / 2.0, Eigen::Vector3d::UnitZ());
  pinocchio::SE3 target_pose = current_pose;
  target_pose.rotation() = current_pose.rotation() * rotation.toRotationMatrix();

  CartesianConfiguration target_config;
  target_config.tform = target_pose;

  FrameTask task("tool0", target_config);

  // Compute error
  Eigen::VectorXd error;
  auto result = task.computeError(*scene_, error);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(error.size(), 6);

  // Position error should be zero
  Eigen::Vector3d position_error = error.head<3>();
  EXPECT_NEAR(position_error.norm(), 0.0, 1e-10);

  // Orientation error should be non-zero
  Eigen::Vector3d orientation_error = error.tail<3>();
  EXPECT_GT(orientation_error.norm(), 0.0);
}

// Test Jacobian computation dimensions
TEST_F(FrameTaskTest, JacobianDimensions) {
  CartesianConfiguration target_pose;
  target_pose.tform = pinocchio::SE3::Identity();

  FrameTask task("tool0", target_pose);

  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, num_variables_);
  auto result = task.computeJacobian(*scene_, jacobian);

  ASSERT_TRUE(result.has_value()) << "computeJacobian failed: " << result.error();
  EXPECT_EQ(jacobian.rows(), 6);
  EXPECT_EQ(jacobian.cols(), num_variables_);
}

// Test Jacobian is not zero
TEST_F(FrameTaskTest, JacobianNonZero) {
  CartesianConfiguration target_pose;
  target_pose.tform = pinocchio::SE3::Identity();

  FrameTask task("tool0", target_pose);

  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, num_variables_);
  auto result = task.computeJacobian(*scene_, jacobian);

  ASSERT_TRUE(result.has_value());

  // Jacobian should not be all zeros
  EXPECT_GT(jacobian.norm(), 0.0);
}

// Test QP objective computation
TEST_F(FrameTaskTest, QpObjectiveComputation) {
  CartesianConfiguration target_pose;
  target_pose.tform = pinocchio::SE3::Identity();

  FrameTaskParams params{.lm_damping = 0.01};
  FrameTask task("tool0", target_pose, params);

  // Compute Jacobian and error
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, num_variables_);
  Eigen::VectorXd error;
  ASSERT_TRUE(task.computeJacobian(*scene_, jacobian).has_value());
  ASSERT_TRUE(task.computeError(*scene_, error).has_value());

  // Compute QP objective matrices
  Eigen::SparseMatrix<double> H(num_variables_, num_variables_);
  Eigen::VectorXd c(num_variables_);
  auto result = task.computeQpObjective(jacobian, error, H, c);

  ASSERT_TRUE(result.has_value());

  // H should be positive semi-definite (diagonal elements >= 0)
  for (int i = 0; i < num_variables_; ++i) {
    EXPECT_GE(H.coeff(i, i), 0.0);
  }

  // H should be symmetric
  Eigen::MatrixXd H_dense = Eigen::MatrixXd(H);
  EXPECT_TRUE(H_dense.isApprox(H_dense.transpose(), 1e-10));
}

// Test invalid frame name
TEST_F(FrameTaskTest, InvalidFrameName) {
  CartesianConfiguration target_pose;
  target_pose.tform = pinocchio::SE3::Identity();

  FrameTask task("nonexistent_frame", target_pose);

  Eigen::VectorXd error;
  auto result = task.computeError(*scene_, error);

  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().find("not found") != std::string::npos);
}

// Test weight matrix effects
TEST_F(FrameTaskTest, WeightMatrixEffects) {
  CartesianConfiguration target_pose;
  target_pose.tform = pinocchio::SE3::Identity();

  // Task with high position cost, low orientation cost
  FrameTaskParams params1{.position_cost = 10.0, .orientation_cost = 0.1};
  FrameTask task1("tool0", target_pose, params1);

  // Task with low position cost, high orientation cost
  FrameTaskParams params2{.position_cost = 0.1, .orientation_cost = 10.0};
  FrameTask task2("tool0", target_pose, params2);

  // Weight matrices should be different
  EXPECT_FALSE(task1.weight.isApprox(task2.weight));

  // First task should weight position errors more
  Eigen::MatrixXd W1 = task1.weight;
  Eigen::MatrixXd W2 = task2.weight;

  // Check that position components (0:3) are weighted differently
  double pos_weight_1 = W1.topLeftCorner(3, 3).trace();
  double pos_weight_2 = W2.topLeftCorner(3, 3).trace();
  EXPECT_GT(pos_weight_1, pos_weight_2);

  // Check that orientation components (3:6) are weighted differently
  double ori_weight_1 = W1.bottomRightCorner(3, 3).trace();
  double ori_weight_2 = W2.bottomRightCorner(3, 3).trace();
  EXPECT_LT(ori_weight_1, ori_weight_2);
}

// Test task gain parameter
TEST_F(FrameTaskTest, TaskGainParameter) {
  CartesianConfiguration target_pose;
  target_pose.tform = pinocchio::SE3::Identity();

  // Create tasks with different gains
  FrameTaskParams params_low{.task_gain = 0.1};
  FrameTask task_low_gain("tool0", target_pose, params_low);

  FrameTaskParams params_high{.task_gain = 0.9};
  FrameTask task_high_gain("tool0", target_pose, params_high);

  EXPECT_LT(task_low_gain.gain, task_high_gain.gain);

  // Gain affects the damping behavior in QP objective
  // Both should compute without error
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, num_variables_);
  Eigen::VectorXd error;
  ASSERT_TRUE(task_low_gain.computeJacobian(*scene_, jacobian).has_value());
  ASSERT_TRUE(task_low_gain.computeError(*scene_, error).has_value());

  Eigen::SparseMatrix<double> H(num_variables_, num_variables_);
  Eigen::VectorXd c(num_variables_);
  auto result = task_low_gain.computeQpObjective(jacobian, error, H, c);
  ASSERT_TRUE(result.has_value());
}

}  // namespace roboplan
