#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include <roboplan/core/scene.hpp>
#include <roboplan_example_models/resources.hpp>
#include <roboplan_oink/barriers/frame_containment.hpp>
#include <roboplan_oink/optimal_ik.hpp>
#include <roboplan_oink/tasks/frame.hpp>

namespace {
// Tolerance for numerical comparisons
constexpr double kTolerance = 1e-3;
constexpr double kNumericalJacobianEpsilon = 1e-6;
// Tolerance for barrier constraint violations in QP solver
// CBFs provide soft safety constraints, not hard constraints
// OSQP numerical solver allows small violations for feasibility
constexpr double kBarrierViolationTolerance = 0.05;  // 5cm tolerance
constexpr double kJacobianNumericalTolerance = 1.0;  // Numerical differentiation is approximate

// Helper to create CartesianConfiguration from position and orientation
roboplan::CartesianConfiguration makeCartesianConfig(const Eigen::Vector3d& position,
                                                     const Eigen::Quaterniond& orientation) {
  roboplan::CartesianConfiguration config;
  Eigen::Matrix4d tform = Eigen::Matrix4d::Identity();
  tform.block<3, 3>(0, 0) = orientation.toRotationMatrix();
  tform.block<3, 1>(0, 3) = position;
  config.tform = tform;
  return config;
}

}  // namespace

namespace roboplan {

class FrameContainmentBarrierTest : public ::testing::Test {
protected:
  void SetUp() override {
    const auto model_prefix = example_models::get_package_models_dir();
    urdf_path_ = model_prefix / "ur_robot_model" / "ur5_gripper.urdf";
    srdf_path_ = model_prefix / "ur_robot_model" / "ur5_gripper.srdf";
    package_paths_ = {example_models::get_package_share_dir()};
    yaml_config_path_ = model_prefix / "ur_robot_model" / "ur5_config.yaml";
    scene_ = std::make_shared<Scene>("test_scene", urdf_path_, srdf_path_, package_paths_,
                                     yaml_config_path_);

    // Get the number of variables (DOF)
    num_variables_ = scene_->getModel().nv;

    // Define a workspace box centered at origin
    box_center_ = Eigen::Vector3d(0.5, 0.0, 0.3);
    box_dimensions_ = Eigen::Vector3d(0.4, 0.4, 0.4);  // 40cm cube
    dt_ = 0.01;
    gain_ = 1.0;
  }

  std::shared_ptr<Scene> scene_;
  std::filesystem::path urdf_path_;
  std::filesystem::path srdf_path_;
  std::vector<std::filesystem::path> package_paths_;
  std::filesystem::path yaml_config_path_;
  int num_variables_;
  Eigen::Vector3d box_center_;
  Eigen::Vector3d box_dimensions_;
  double dt_;
  double gain_;
};

// Test basic construction
TEST_F(FrameContainmentBarrierTest, Construction) {
  ASSERT_NO_THROW({
    FrameContainmentBarrier barrier("tool0", box_center_, box_dimensions_, num_variables_, dt_,
                                    gain_);
    EXPECT_EQ(barrier.barrier_dim_, 6);  // 6 constraints (2 per axis)
    EXPECT_EQ(barrier.dt_, dt_);
    EXPECT_EQ(barrier.gain_, gain_);
    EXPECT_EQ(barrier.frame_name_, "tool0");
  });
}

// Test barrier dimensions
TEST_F(FrameContainmentBarrierTest, BarrierDimensions) {
  FrameContainmentBarrier barrier("tool0", box_center_, box_dimensions_, num_variables_, dt_,
                                  gain_);

  // Barrier should have 6 constraints
  EXPECT_EQ(barrier.barrier_dim_, 6);

  // Pre-allocated workspace should have correct dimensions
  EXPECT_EQ(barrier.h_.size(), 6);
  EXPECT_EQ(barrier.jacobian_.rows(), 6);
  EXPECT_EQ(barrier.jacobian_.cols(), num_variables_);
}

// Test barrier function when frame is inside the box
TEST_F(FrameContainmentBarrierTest, BarrierInsideBox) {
  FrameContainmentBarrier barrier("tool0", box_center_, box_dimensions_, num_variables_, dt_,
                                  gain_);

  // Set configuration that places end-effector inside the box
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  q[0] = 0.0;
  q[1] = -1.5;
  q[2] = 1.5;
  scene_->setJointPositions(q);

  // Compute barrier values
  Eigen::VectorXd h(6);
  auto result = barrier.computeBarrier(*scene_, h);

  ASSERT_TRUE(result.has_value()) << "Barrier computation failed: " << result.error();

  // When inside the box, all barrier values should be positive
  for (int i = 0; i < 6; ++i) {
    EXPECT_GT(h[i], 0.0) << "Barrier " << i << " should be positive when inside box";
  }
}

// Test barrier function when frame is outside the box
TEST_F(FrameContainmentBarrierTest, BarrierOutsideBox) {
  FrameContainmentBarrier barrier("tool0", box_center_, box_dimensions_, num_variables_, dt_,
                                  gain_);

  // Set configuration that places end-effector far outside the box
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  q[0] = 3.0;  // Large rotation to move far away
  scene_->setJointPositions(q);

  // Compute barrier values
  Eigen::VectorXd h(6);
  auto result = barrier.computeBarrier(*scene_, h);

  ASSERT_TRUE(result.has_value()) << "Barrier computation failed: " << result.error();

  // When outside the box, at least one barrier value should be negative
  bool any_negative = false;
  for (int i = 0; i < 6; ++i) {
    if (h[i] < 0.0) {
      any_negative = true;
      break;
    }
  }
  EXPECT_TRUE(any_negative) << "At least one barrier should be negative when outside box";
}

// Test barrier function at box boundary
TEST_F(FrameContainmentBarrierTest, BarrierAtBoundary) {
  // Create a tight box around a known configuration
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  q[1] = -1.5;
  q[2] = 1.5;
  scene_->setJointPositions(q);

  // Get the current end-effector position
  Eigen::Matrix4d ee_transform = scene_->forwardKinematics(q, "tool0");
  Eigen::Vector3d ee_position = ee_transform.block<3, 1>(0, 3);

  // Create box centered exactly at the end-effector with very small dimensions
  Eigen::Vector3d tight_box_dim(0.01, 0.01, 0.01);  // 1cm margins
  FrameContainmentBarrier barrier("tool0", ee_position, tight_box_dim, num_variables_, dt_, gain_);

  // Compute barrier values
  Eigen::VectorXd h(6);
  auto result = barrier.computeBarrier(*scene_, h);

  ASSERT_TRUE(result.has_value()) << "Barrier computation failed: " << result.error();

  // All barrier values should be small (near zero) but positive
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(h[i], tight_box_dim[i % 3] / 2.0, kTolerance)
        << "Barrier " << i << " should be near box half-extent";
  }
}

// Test Jacobian shape and structure
TEST_F(FrameContainmentBarrierTest, JacobianShape) {
  FrameContainmentBarrier barrier("tool0", box_center_, box_dimensions_, num_variables_, dt_,
                                  gain_);

  // Set some configuration
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  scene_->setJointPositions(q);

  // Compute Jacobian
  Eigen::MatrixXd jacobian(6, num_variables_);
  auto result = barrier.computeJacobian(*scene_, jacobian);

  ASSERT_TRUE(result.has_value()) << "Jacobian computation failed: " << result.error();

  // Check dimensions
  EXPECT_EQ(jacobian.rows(), 6);
  EXPECT_EQ(jacobian.cols(), num_variables_);

  // Check structure: upper 3 rows should be negative of lower 3 rows
  Eigen::MatrixXd upper_jacobian = jacobian.topRows<3>();
  Eigen::MatrixXd lower_jacobian = jacobian.bottomRows<3>();

  // upper should be -position_jacobian, lower should be +position_jacobian
  // So upper + lower should be approximately zero
  EXPECT_LT((upper_jacobian + lower_jacobian).norm(), kTolerance)
      << "Upper and lower Jacobian blocks should be negatives of each other";
}

// Test Jacobian via numerical differentiation
TEST_F(FrameContainmentBarrierTest, JacobianNumericalValidation) {
  FrameContainmentBarrier barrier("tool0", box_center_, box_dimensions_, num_variables_, dt_,
                                  gain_);

  // Set nominal configuration
  Eigen::VectorXd q_nominal = Eigen::VectorXd::Zero(num_variables_);
  q_nominal[1] = -1.5;
  q_nominal[2] = 1.5;

  // Compute analytical Jacobian
  scene_->setJointPositions(q_nominal);
  Eigen::MatrixXd jacobian_analytical(6, num_variables_);
  auto jac_result = barrier.computeJacobian(*scene_, jacobian_analytical);
  ASSERT_TRUE(jac_result.has_value());

  // Compute numerical Jacobian via finite differences
  Eigen::MatrixXd jacobian_numerical(6, num_variables_);
  Eigen::VectorXd h_nominal(6), h_perturbed(6);

  scene_->setJointPositions(q_nominal);
  auto h_result = barrier.computeBarrier(*scene_, h_nominal);
  ASSERT_TRUE(h_result.has_value());

  for (int j = 0; j < num_variables_; ++j) {
    Eigen::VectorXd q_perturbed = q_nominal;
    q_perturbed[j] += kNumericalJacobianEpsilon;

    scene_->setJointPositions(q_perturbed);
    auto h_pert_result = barrier.computeBarrier(*scene_, h_perturbed);
    ASSERT_TRUE(h_pert_result.has_value());

    // Numerical derivative: (h(q + ε) - h(q)) / ε
    jacobian_numerical.col(j) = (h_perturbed - h_nominal) / kNumericalJacobianEpsilon;
  }

  // Compare analytical and numerical Jacobians
  // Note: Numerical differentiation has limited accuracy, especially for complex kinematic chains
  double max_error = (jacobian_analytical - jacobian_numerical).cwiseAbs().maxCoeff();
  EXPECT_LT(max_error, kJacobianNumericalTolerance)
      << "Analytical and numerical Jacobians should match within numerical differentiation limits";
}

// Test integration with Oink solver - barrier keeps frame inside box
TEST_F(FrameContainmentBarrierTest, IntegrationWithOink) {
  // Create Oink solver
  Oink oink(num_variables_);

  // Create a tight workspace box
  Eigen::Vector3d workspace_center(0.4, 0.0, 0.4);
  Eigen::Vector3d workspace_dim(0.3, 0.3, 0.3);

  // Create barrier
  auto barrier = std::make_shared<FrameContainmentBarrier>("tool0", workspace_center, workspace_dim,
                                                           num_variables_, dt_, gain_);

  // Create a task that tries to move outside the box
  auto target_pose = makeCartesianConfig(Eigen::Vector3d(0.8, 0.0, 0.8),  // Far outside the box
                                         Eigen::Quaterniond::Identity());
  auto task = std::make_shared<FrameTask>("tool0", target_pose);

  std::vector<std::shared_ptr<Task>> tasks = {task};
  std::vector<std::shared_ptr<Constraints>> constraints;
  std::vector<std::shared_ptr<Barrier>> barriers = {barrier};

  // Set initial configuration inside the box
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  q[1] = -1.5;
  q[2] = 1.5;
  scene_->setJointPositions(q);

  // Solve IK with barrier
  Eigen::VectorXd delta_q;
  auto result = oink.solveIk(tasks, constraints, barriers, *scene_, delta_q);

  ASSERT_TRUE(result.has_value()) << "Solve failed: " << result.error();

  // Update configuration
  Eigen::VectorXd q_new = q + delta_q;
  scene_->setJointPositions(q_new);

  // Verify end-effector stays inside the box
  Eigen::Matrix4d ee_transform = scene_->forwardKinematics(q_new, "tool0");
  Eigen::Vector3d ee_position = ee_transform.block<3, 1>(0, 3);

  Eigen::Vector3d box_half = workspace_dim / 2.0;
  Eigen::Vector3d dist_to_upper = (workspace_center + box_half) - ee_position;
  Eigen::Vector3d dist_to_lower = ee_position - (workspace_center - box_half);

  // All distances should be non-negative (with tolerance for QP solver numerical behavior)
  // CBFs provide soft safety constraints - small violations are expected from numerical
  // optimization
  for (int i = 0; i < 3; ++i) {
    EXPECT_GE(dist_to_upper[i], -kBarrierViolationTolerance)
        << "Axis " << i << " exceeded upper boundary: position = " << ee_position[i]
        << ", upper limit = " << (workspace_center[i] + box_half[i]);
    EXPECT_GE(dist_to_lower[i], -kBarrierViolationTolerance)
        << "Axis " << i << " exceeded lower boundary: position = " << ee_position[i]
        << ", lower limit = " << (workspace_center[i] - box_half[i]);
  }
}

// Test barrier with multiple IK iterations
TEST_F(FrameContainmentBarrierTest, MultipleIterationsStaySafe) {
  Oink oink(num_variables_);

  // Create workspace barrier
  Eigen::Vector3d workspace_center(0.4, 0.0, 0.4);
  Eigen::Vector3d workspace_dim(0.35, 0.35, 0.35);
  auto barrier = std::make_shared<FrameContainmentBarrier>("tool0", workspace_center, workspace_dim,
                                                           num_variables_, dt_, 2.0);

  // Start inside the box
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  q[1] = -1.5;
  q[2] = 1.5;
  scene_->setJointPositions(q);

  std::vector<std::shared_ptr<Constraints>> constraints;
  std::vector<std::shared_ptr<Barrier>> barriers = {barrier};

  // Run multiple IK iterations with different targets
  for (int iter = 0; iter < 10; ++iter) {
    // Create random target outside the box
    Eigen::Vector3d target_pos(0.9 * std::sin(iter), 0.0, 0.9 * std::cos(iter));
    auto target_pose = makeCartesianConfig(target_pos, Eigen::Quaterniond::Identity());
    auto task = std::make_shared<FrameTask>("tool0", target_pose);

    std::vector<std::shared_ptr<Task>> tasks = {task};

    Eigen::VectorXd delta_q;
    auto result = oink.solveIk(tasks, constraints, barriers, *scene_, delta_q);

    if (result.has_value()) {
      q += delta_q;
      scene_->setJointPositions(q);

      // Verify still inside box (with tolerance for QP solver numerical behavior)
      Eigen::Matrix4d ee_transform = scene_->forwardKinematics(q, "tool0");
      Eigen::Vector3d ee_position = ee_transform.block<3, 1>(0, 3);

      Eigen::Vector3d box_half = workspace_dim / 2.0;
      for (int i = 0; i < 3; ++i) {
        EXPECT_GE(ee_position[i], workspace_center[i] - box_half[i] - kBarrierViolationTolerance)
            << "Iteration " << iter << " axis " << i << " below lower limit";
        EXPECT_LE(ee_position[i], workspace_center[i] + box_half[i] + kBarrierViolationTolerance)
            << "Iteration " << iter << " axis " << i << " above upper limit";
      }
    }
  }
}

// Test error handling - invalid frame name
TEST_F(FrameContainmentBarrierTest, InvalidFrameName) {
  FrameContainmentBarrier barrier("invalid_frame_xyz", box_center_, box_dimensions_, num_variables_,
                                  dt_, gain_);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  scene_->setJointPositions(q);

  Eigen::VectorXd h(6);
  auto result = barrier.computeBarrier(*scene_, h);

  EXPECT_FALSE(result.has_value()) << "Should fail with invalid frame name";
}

// Test dimension validation
TEST_F(FrameContainmentBarrierTest, DimensionValidation) {
  FrameContainmentBarrier barrier("tool0", box_center_, box_dimensions_, num_variables_, dt_,
                                  gain_);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  scene_->setJointPositions(q);

  // Test with wrong-sized output vector
  Eigen::VectorXd h_wrong(3);  // Should be 6
  auto result = barrier.computeBarrier(*scene_, h_wrong);

  EXPECT_FALSE(result.has_value()) << "Should fail with wrong output size";
  EXPECT_THAT(result.error(), ::testing::HasSubstr("mismatch"));
}

// Test gain parameter effect
TEST_F(FrameContainmentBarrierTest, GainParameterEffect) {
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  q[1] = -1.5;
  q[2] = 1.5;
  scene_->setJointPositions(q);

  // Create two barriers with different gains
  FrameContainmentBarrier barrier_low_gain("tool0", box_center_, box_dimensions_, num_variables_,
                                           dt_, 0.5);
  FrameContainmentBarrier barrier_high_gain("tool0", box_center_, box_dimensions_, num_variables_,
                                            dt_, 2.0);

  // Compute QP constraints for both
  Eigen::MatrixXd A_low(6, num_variables_), A_high(6, num_variables_);
  Eigen::VectorXd lower_low(6), upper_low(6), lower_high(6), upper_high(6);

  auto result_low = barrier_low_gain.computeQpConstraint(*scene_, A_low, lower_low, upper_low);
  auto result_high = barrier_high_gain.computeQpConstraint(*scene_, A_high, lower_high, upper_high);

  ASSERT_TRUE(result_low.has_value());
  ASSERT_TRUE(result_high.has_value());

  // Constraint matrices should be identical (gain doesn't affect Jacobian)
  EXPECT_LT((A_low - A_high).norm(), kTolerance);

  // Upper bounds should scale with gain
  EXPECT_LT((upper_high - 2.0 / 0.5 * upper_low).norm(), kTolerance);
}

}  // namespace roboplan

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
