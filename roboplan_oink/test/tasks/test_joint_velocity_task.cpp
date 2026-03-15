#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>

#include <roboplan/core/scene.hpp>
#include <roboplan_example_models/resources.hpp>
#include <roboplan_oink/tasks/joint_velocity.hpp>

namespace roboplan {

class JointVelocityTaskTest : public ::testing::Test {
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
    nq_ = model.nq;
    nv_ = model.nv;

    // Set a non-zero initial configuration
    Eigen::VectorXd q = Eigen::VectorXd::Zero(nq_);
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
  int nq_;
  int nv_;
};

// Test joint velocity task construction
TEST_F(JointVelocityTaskTest, Construction) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_);

  // Test default construction
  JointVelocityTask task1(target_delta_q, joint_weights);
  EXPECT_EQ(task1.target_delta_q.size(), nv_);
  EXPECT_EQ(task1.joint_weights.size(), nv_);
  EXPECT_EQ(task1.gain, 1.0);
  EXPECT_EQ(task1.lm_damping, 0.0);

  // Test construction with custom options
  JointVelocityTaskOptions options{
      .task_gain = 0.8,
      .lm_damping = 0.01,
  };
  JointVelocityTask task2(target_delta_q, joint_weights, options);
  EXPECT_EQ(task2.gain, 0.8);
  EXPECT_EQ(task2.lm_damping, 0.01);
}

// Test that negative joint weights throw
TEST_F(JointVelocityTaskTest, NegativeWeightThrows) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_);
  joint_weights(2) = -1.0;  // Negative weight

  EXPECT_THROW({ JointVelocityTask task(target_delta_q, joint_weights); }, std::invalid_argument);
}

// Test that mismatched sizes throw
TEST_F(JointVelocityTaskTest, MismatchedSizeThrows) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_ + 1);  // Different size

  EXPECT_THROW({ JointVelocityTask task(target_delta_q, joint_weights); }, std::invalid_argument);
}

// Test error computation with zero target (should be zero error)
TEST_F(JointVelocityTaskTest, ErrorWithZeroTarget) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_);

  JointVelocityTask task(target_delta_q, joint_weights);

  auto result = task.computeError(*scene_);

  ASSERT_TRUE(result.has_value()) << "computeError failed: " << result.error();
  EXPECT_EQ(task.error_container.size(), nv_);

  // Error should be zero (error = target_delta_q = 0)
  EXPECT_NEAR(task.error_container.norm(), 0.0, 1e-10);
}

// Test error computation with non-zero target
TEST_F(JointVelocityTaskTest, ErrorWithNonZeroTarget) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);
  target_delta_q(0) = 0.5;
  target_delta_q(1) = -0.3;
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_);

  JointVelocityTask task(target_delta_q, joint_weights);

  auto result = task.computeError(*scene_);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(task.error_container.size(), nv_);

  // Error equals target_delta_q (so that QP solution is dq = gain * target_delta_q)
  EXPECT_NEAR(task.error_container(0), 0.5, 1e-10);
  EXPECT_NEAR(task.error_container(1), -0.3, 1e-10);

  // Other joints should have zero error
  for (int i = 2; i < nv_; ++i) {
    EXPECT_NEAR(task.error_container(i), 0.0, 1e-10);
  }
}

// Test Jacobian computation dimensions and identity
TEST_F(JointVelocityTaskTest, JacobianIsNegativeIdentity) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_);

  JointVelocityTask task(target_delta_q, joint_weights);

  auto result = task.computeJacobian(*scene_);

  ASSERT_TRUE(result.has_value()) << "computeJacobian failed: " << result.error();
  EXPECT_EQ(task.jacobian_container.rows(), nv_);
  EXPECT_EQ(task.jacobian_container.cols(), nv_);

  // Jacobian should be negative identity
  Eigen::MatrixXd expected = -Eigen::MatrixXd::Identity(nv_, nv_);
  EXPECT_TRUE(task.jacobian_container.isApprox(expected, 1e-6))
      << "Jacobian is not negative identity" << std::endl
      << "Computed:\n"
      << task.jacobian_container << std::endl
      << "Expected:\n"
      << expected << std::endl;
}

// Test QP objective computation
TEST_F(JointVelocityTaskTest, QpObjectiveComputation) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_);

  JointVelocityTaskOptions options{.lm_damping = 0.01};
  JointVelocityTask task(target_delta_q, joint_weights, options);

  // Compute QP objective matrices
  Eigen::SparseMatrix<double> H(nv_, nv_);
  Eigen::VectorXd c(nv_);
  auto result = task.computeQpObjective(*scene_, H, c);

  ASSERT_TRUE(result.has_value());

  // H should be positive semi-definite (diagonal elements >= 0)
  for (int i = 0; i < nv_; ++i) {
    EXPECT_GE(H.coeff(i, i), 0.0);
  }

  // H should be symmetric
  Eigen::MatrixXd H_dense = Eigen::MatrixXd(H);
  EXPECT_TRUE(H_dense.isApprox(H_dense.transpose(), 1e-10));
}

// Test weight matrix with per-joint weights
TEST_F(JointVelocityTaskTest, WeightMatrixPerJoint) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);

  // Create non-uniform weights
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Zero(nv_);
  joint_weights(0) = 10.0;  // High weight on first joint
  joint_weights(1) = 1.0;   // Normal weight on second
  joint_weights(2) = 0.1;   // Low weight on third
  // Remaining joints have zero weight

  JointVelocityTask task(target_delta_q, joint_weights);

  // Check weight matrix diagonal
  const Eigen::MatrixXd& W = task.weight;
  EXPECT_EQ(W.rows(), nv_);
  EXPECT_EQ(W.cols(), nv_);

  // Verify diagonal elements: sqrt(joint_weight)
  EXPECT_NEAR(W(0, 0), std::sqrt(10.0), 1e-10);
  EXPECT_NEAR(W(1, 1), std::sqrt(1.0), 1e-10);
  EXPECT_NEAR(W(2, 2), std::sqrt(0.1), 1e-10);
  EXPECT_NEAR(W(3, 3), 0.0, 1e-10);  // Zero weight

  // Off-diagonal should be zero
  for (int i = 0; i < nv_; ++i) {
    for (int j = 0; j < nv_; ++j) {
      if (i != j) {
        EXPECT_NEAR(W(i, j), 0.0, 1e-10);
      }
    }
  }
}

// Test that the QP solution minimizes ||dq - target_delta_q||^2
// The analytical solution is dq = target_delta_q (when gain=1, no damping)
TEST_F(JointVelocityTaskTest, AnalyticalSolutionMatchesTarget) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);
  target_delta_q(0) = 0.5;
  target_delta_q(1) = -0.3;
  target_delta_q(2) = 0.1;
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_);

  JointVelocityTaskOptions options{.task_gain = 1.0, .lm_damping = 0.0};
  JointVelocityTask task(target_delta_q, joint_weights, options);

  // Compute error and Jacobian
  auto error_result = task.computeError(*scene_);
  ASSERT_TRUE(error_result.has_value());

  auto jacobian_result = task.computeJacobian(*scene_);
  ASSERT_TRUE(jacobian_result.has_value());

  // For the QP objective: min ||J*dq + gain*e||^2
  // With J = -I, e = target_delta_q, gain = 1:
  //   min ||(-I)*dq + target_delta_q||^2 = ||-dq + target_delta_q||^2
  // Minimized when: -dq + target_delta_q = 0 => dq = target_delta_q
  //
  // The analytical solution is: dq = gain * (-J)^{-1} * e = gain * I * e = gain * e
  Eigen::VectorXd dq_analytical =
      task.gain * (-task.jacobian_container).inverse() * task.error_container;

  EXPECT_TRUE(dq_analytical.isApprox(target_delta_q, 1e-10))
      << "Analytical solution should match target_delta_q" << std::endl
      << "dq_analytical: " << dq_analytical.transpose() << std::endl
      << "target_delta_q: " << target_delta_q.transpose() << std::endl;
}

// Test invalid target_delta_q size
TEST_F(JointVelocityTaskTest, InvalidTargetSize) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_ + 1);  // Wrong size
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_ + 1);   // Match to avoid constructor throw

  JointVelocityTask task(target_delta_q, joint_weights);

  auto result = task.computeError(*scene_);

  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().find("size") != std::string::npos);
}

// Test invalid joint_weights size in computeJacobian
TEST_F(JointVelocityTaskTest, InvalidWeightsSizeInJacobian) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_ + 1);  // Wrong size
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_ + 1);   // Match to avoid constructor throw

  JointVelocityTask task(target_delta_q, joint_weights);

  auto result = task.computeJacobian(*scene_);

  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().find("size") != std::string::npos);
}

// Test task gain parameter affects the solution
TEST_F(JointVelocityTaskTest, TaskGainScalesSolution) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);
  target_delta_q(0) = 1.0;
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_);

  // With gain=0.5, the solution should be dq = 0.5 * target_delta_q
  JointVelocityTaskOptions options{.task_gain = 0.5, .lm_damping = 0.0};
  JointVelocityTask task(target_delta_q, joint_weights, options);

  auto error_result = task.computeError(*scene_);
  ASSERT_TRUE(error_result.has_value());

  auto jacobian_result = task.computeJacobian(*scene_);
  ASSERT_TRUE(jacobian_result.has_value());

  // dq = gain * e (where e = target_delta_q)
  Eigen::VectorXd dq_analytical =
      task.gain * (-task.jacobian_container).inverse() * task.error_container;

  // With gain=0.5, dq should be 0.5 * target_delta_q
  Eigen::VectorXd expected = 0.5 * target_delta_q;
  EXPECT_TRUE(dq_analytical.isApprox(expected, 1e-10))
      << "Solution should be gain * target_delta_q" << std::endl
      << "dq_analytical: " << dq_analytical.transpose() << std::endl
      << "expected: " << expected.transpose() << std::endl;
}

// Test that this task is independent of robot configuration
// (unlike ConfigurationTask which depends on current q)
TEST_F(JointVelocityTaskTest, IndependentOfConfiguration) {
  Eigen::VectorXd target_delta_q = Eigen::VectorXd::Zero(nv_);
  target_delta_q(0) = 0.5;
  Eigen::VectorXd joint_weights = Eigen::VectorXd::Ones(nv_);

  JointVelocityTask task(target_delta_q, joint_weights);

  // Compute error at initial configuration
  auto result1 = task.computeError(*scene_);
  ASSERT_TRUE(result1.has_value());
  Eigen::VectorXd error1 = task.error_container;

  // Change configuration
  Eigen::VectorXd new_q = Eigen::VectorXd::Constant(nq_, 1.5);
  scene_->setJointPositions(new_q);

  // Compute error at new configuration
  auto result2 = task.computeError(*scene_);
  ASSERT_TRUE(result2.has_value());
  Eigen::VectorXd error2 = task.error_container;

  // Errors should be identical (task doesn't depend on q)
  EXPECT_TRUE(error1.isApprox(error2, 1e-10))
      << "Error should be independent of robot configuration";
}

}  // namespace roboplan
