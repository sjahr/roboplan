#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include <roboplan/core/scene.hpp>
#include <roboplan_example_models/resources.hpp>
#include <roboplan_oink/constraints/position_limit.hpp>
#include <roboplan_oink/constraints/velocity_limit.hpp>
#include <roboplan_oink/optimal_ik.hpp>
#include <roboplan_oink/tasks/frame.hpp>

namespace {
// Tolerance for OSQP constraint satisfaction
// OSQP is a numerical solver with finite precision, so we allow small violations (~1e-4)
constexpr double kTolerance = 1e-3;

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

class OinkTest : public ::testing::Test {
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
  }

  std::shared_ptr<Scene> scene_;
  std::filesystem::path urdf_path_;
  std::filesystem::path srdf_path_;
  std::vector<std::filesystem::path> package_paths_;
  std::filesystem::path yaml_config_path_;
  int num_variables_;
};

// Test basic construction and initialization
TEST_F(OinkTest, Construction) {
  ASSERT_NO_THROW({
    Oink oink(num_variables_);
    EXPECT_EQ(oink.num_variables, num_variables_);
    EXPECT_EQ(oink.last_constraint_rows, -1);  // -1 = uninitialized
  });
}

// Test solving with no constraints
TEST_F(OinkTest, SolveWithNoConstraints) {
  Oink oink(num_variables_);

  // Set initial configuration
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  scene_->setJointPositions(q);

  // Create a simple frame task (move end effector to target)
  auto target_pose =
      makeCartesianConfig(Eigen::Vector3d(0.3, 0.2, 0.5), Eigen::Quaterniond::Identity());

  auto task = std::make_shared<FrameTask>("tool0", target_pose);
  std::vector<std::shared_ptr<Task>> tasks = {task};
  std::vector<std::shared_ptr<Constraints>> constraints;
  std::vector<std::shared_ptr<Barrier>> barriers;

  Eigen::VectorXd delta_q;
  auto result = oink.solveIk(tasks, constraints, barriers, *scene_, delta_q);

  ASSERT_TRUE(result.has_value()) << "Solve failed: " << result.error();
  EXPECT_EQ(delta_q.size(), num_variables_);

  // Verify delta_q is not all zeros (should have moved)
  EXPECT_GT(delta_q.norm(), kTolerance);
}

// Test solving with velocity constraints
TEST_F(OinkTest, SolveWithVelocityConstraints) {
  Oink oink(num_variables_);

  // Set initial configuration
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  scene_->setJointPositions(q);

  // Create velocity limits
  double dt = 0.01;                                                     // 10ms timestep
  Eigen::VectorXd v_max = Eigen::VectorXd::Ones(num_variables_) * 1.0;  // 1 rad/s max
  auto vel_constraint = std::make_shared<VelocityLimit>(num_variables_, dt, v_max);

  // Create a frame task
  auto target_pose =
      makeCartesianConfig(Eigen::Vector3d(0.3, 0.2, 0.5), Eigen::Quaterniond::Identity());
  auto task = std::make_shared<FrameTask>("tool0", target_pose);
  std::vector<std::shared_ptr<Task>> tasks = {task};
  std::vector<std::shared_ptr<Constraints>> constraints = {vel_constraint};

  Eigen::VectorXd delta_q;
  std::vector<std::shared_ptr<Barrier>> barriers;
  auto result = oink.solveIk(tasks, constraints, barriers, *scene_, delta_q);

  ASSERT_TRUE(result.has_value()) << "Solve failed: " << result.error();
  EXPECT_EQ(delta_q.size(), num_variables_);

  // Verify velocity constraints are satisfied: |dq| <= dt * v_max
  for (int i = 0; i < num_variables_; ++i) {
    EXPECT_LE(std::abs(delta_q[i]), dt * v_max[i] + kTolerance)
        << "Joint " << i << " violated velocity constraint";
  }
}

// Test solving with position constraints
TEST_F(OinkTest, SolveWithPositionConstraints) {
  Oink oink(num_variables_);

  // Set initial configuration near joint limits
  const auto& model = scene_->getModel();
  Eigen::VectorXd q = 0.9 * model.upperPositionLimit;
  scene_->setJointPositions(q);

  // Create position limit constraint
  double gain = 0.5;
  auto pos_constraint = std::make_shared<PositionLimit>(num_variables_, gain);

  // Create a task that would push toward the limit
  auto target_pose =
      makeCartesianConfig(Eigen::Vector3d(0.8, 0.0, 0.5), Eigen::Quaterniond::Identity());
  auto task = std::make_shared<FrameTask>("tool0", target_pose);
  std::vector<std::shared_ptr<Task>> tasks = {task};
  std::vector<std::shared_ptr<Constraints>> constraints = {pos_constraint};

  Eigen::VectorXd delta_q;
  std::vector<std::shared_ptr<Barrier>> barriers;
  auto result = oink.solveIk(tasks, constraints, barriers, *scene_, delta_q);

  ASSERT_TRUE(result.has_value()) << "Solve failed: " << result.error();
  EXPECT_EQ(delta_q.size(), num_variables_);

  // Verify we don't exceed joint limits
  Eigen::VectorXd q_new = q + delta_q;
  for (int i = 0; i < num_variables_; ++i) {
    if (std::isfinite(model.lowerPositionLimit[i])) {
      EXPECT_GE(q_new[i], model.lowerPositionLimit[i] - kTolerance)
          << "Joint " << i << " went below lower limit";
    }
    if (std::isfinite(model.upperPositionLimit[i])) {
      EXPECT_LE(q_new[i], model.upperPositionLimit[i] + kTolerance)
          << "Joint " << i << " went above upper limit";
    }
  }
}

// Test solving with multiple constraints
TEST_F(OinkTest, SolveWithMultipleConstraints) {
  Oink oink(num_variables_);

  // Set initial configuration
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  scene_->setJointPositions(q);

  // Create both velocity and position constraints
  double dt = 0.01;
  Eigen::VectorXd v_max = Eigen::VectorXd::Ones(num_variables_) * 2.0;
  auto vel_constraint = std::make_shared<VelocityLimit>(num_variables_, dt, v_max);
  auto pos_constraint = std::make_shared<PositionLimit>(num_variables_, 0.8);

  // Create a frame task
  auto target_pose =
      makeCartesianConfig(Eigen::Vector3d(0.4, 0.1, 0.6), Eigen::Quaterniond::Identity());
  auto task = std::make_shared<FrameTask>("tool0", target_pose);
  std::vector<std::shared_ptr<Task>> tasks = {task};
  std::vector<std::shared_ptr<Constraints>> constraints = {vel_constraint, pos_constraint};

  Eigen::VectorXd delta_q;
  std::vector<std::shared_ptr<Barrier>> barriers;
  auto result = oink.solveIk(tasks, constraints, barriers, *scene_, delta_q);

  ASSERT_TRUE(result.has_value()) << "Solve failed: " << result.error();
  EXPECT_EQ(delta_q.size(), num_variables_);

  // Verify velocity constraints
  for (int i = 0; i < num_variables_; ++i) {
    EXPECT_LE(std::abs(delta_q[i]), dt * v_max[i] + kTolerance);
  }
}

// Test workspace caching - solve twice to ensure no reallocation on second call
TEST_F(OinkTest, WorkspaceCaching) {
  Oink oink(num_variables_);

  // Set initial configuration
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  scene_->setJointPositions(q);

  // Create constraints
  double dt = 0.01;
  Eigen::VectorXd v_max = Eigen::VectorXd::Ones(num_variables_) * 1.0;
  auto vel_constraint = std::make_shared<VelocityLimit>(num_variables_, dt, v_max);

  // Create task
  auto target_pose =
      makeCartesianConfig(Eigen::Vector3d(0.3, 0.2, 0.5), Eigen::Quaterniond::Identity());
  auto task = std::make_shared<FrameTask>("tool0", target_pose);
  std::vector<std::shared_ptr<Task>> tasks = {task};
  std::vector<std::shared_ptr<Constraints>> constraints = {vel_constraint};

  // First solve - workspace allocation
  std::vector<std::shared_ptr<Barrier>> barriers;
  Eigen::VectorXd delta_q1;
  auto result1 = oink.solveIk(tasks, constraints, barriers, *scene_, delta_q1);
  ASSERT_TRUE(result1.has_value()) << "First solve failed: " << result1.error();

  // Verify workspace dimensions
  EXPECT_EQ(oink.constraint_workspace_A.rows(), num_variables_);
  EXPECT_EQ(oink.constraint_workspace_A.cols(), num_variables_);
  EXPECT_EQ(oink.constraint_workspace_lower.size(), num_variables_);
  EXPECT_EQ(oink.constraint_workspace_upper.size(), num_variables_);
  EXPECT_EQ(oink.last_constraint_rows, num_variables_);

  // Store matrix data pointers to verify no reallocation
  const double* A_data_ptr = oink.constraint_workspace_A.data();
  const double* lower_data_ptr = oink.constraint_workspace_lower.data();
  const double* upper_data_ptr = oink.constraint_workspace_upper.data();

  // Second solve - should reuse workspace (no allocation)
  q += delta_q1;  // Update configuration
  scene_->setJointPositions(q);

  Eigen::VectorXd delta_q2;
  auto result2 = oink.solveIk(tasks, constraints, barriers, *scene_, delta_q2);
  ASSERT_TRUE(result2.has_value()) << "Second solve failed: " << result2.error();

  // Verify workspace was reused (same pointers)
  EXPECT_EQ(oink.constraint_workspace_A.data(), A_data_ptr) << "Workspace A was reallocated!";
  EXPECT_EQ(oink.constraint_workspace_lower.data(), lower_data_ptr)
      << "Workspace lower was reallocated!";
  EXPECT_EQ(oink.constraint_workspace_upper.data(), upper_data_ptr)
      << "Workspace upper was reallocated!";
  EXPECT_EQ(oink.last_constraint_rows, num_variables_);
}

// Test constraint dimension validation
TEST_F(OinkTest, ConstraintDimensionValidation) {
  // Create velocity constraint with WRONG size - should throw at construction
  double dt = 0.01;
  Eigen::VectorXd v_max_wrong = Eigen::VectorXd::Ones(num_variables_ - 1);  // Wrong size!

  // Constructor should throw std::invalid_argument due to size mismatch
  EXPECT_THROW(
      { auto vel_constraint = std::make_shared<VelocityLimit>(num_variables_, dt, v_max_wrong); },
      std::invalid_argument);
}

// Test solving with dynamically changing constraint count
TEST_F(OinkTest, DynamicConstraintCount) {
  Oink oink(num_variables_);

  // Set initial configuration
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  scene_->setJointPositions(q);

  // Create task
  auto target_pose =
      makeCartesianConfig(Eigen::Vector3d(0.3, 0.2, 0.5), Eigen::Quaterniond::Identity());
  auto task = std::make_shared<FrameTask>("tool0", target_pose);
  std::vector<std::shared_ptr<Task>> tasks = {task};

  // First solve with one constraint
  double dt = 0.01;
  Eigen::VectorXd v_max = Eigen::VectorXd::Ones(num_variables_) * 1.0;
  auto vel_constraint = std::make_shared<VelocityLimit>(num_variables_, dt, v_max);
  std::vector<std::shared_ptr<Constraints>> constraints1 = {vel_constraint};
  std::vector<std::shared_ptr<Barrier>> barriers;

  Eigen::VectorXd delta_q1;
  auto result1 = oink.solveIk(tasks, constraints1, barriers, *scene_, delta_q1);
  ASSERT_TRUE(result1.has_value()) << "First solve failed: " << result1.error();
  EXPECT_EQ(oink.last_constraint_rows, num_variables_);

  // Second solve with two constraints (workspace should resize)
  auto pos_constraint = std::make_shared<PositionLimit>(num_variables_, 0.8);
  std::vector<std::shared_ptr<Constraints>> constraints2 = {vel_constraint, pos_constraint};

  Eigen::VectorXd delta_q2;
  auto result2 = oink.solveIk(tasks, constraints2, barriers, *scene_, delta_q2);
  ASSERT_TRUE(result2.has_value()) << "Second solve failed: " << result2.error();
  EXPECT_EQ(oink.last_constraint_rows, 2 * num_variables_);

  // Verify workspace dimensions grew
  EXPECT_EQ(oink.constraint_workspace_A.rows(), 2 * num_variables_);
  EXPECT_EQ(oink.constraint_workspace_lower.size(), 2 * num_variables_);
  EXPECT_EQ(oink.constraint_workspace_upper.size(), 2 * num_variables_);

  // Third solve back to one constraint (workspace should resize down)
  Eigen::VectorXd delta_q3;
  auto result3 = oink.solveIk(tasks, constraints1, barriers, *scene_, delta_q3);
  ASSERT_TRUE(result3.has_value()) << "Third solve failed: " << result3.error();
  EXPECT_EQ(oink.last_constraint_rows, num_variables_);
}

// Test Eigen::Ref safety - verify constraints cannot resize views
TEST_F(OinkTest, EigenRefSafety) {
  Oink oink(num_variables_);

  // Set initial configuration
  Eigen::VectorXd q = Eigen::VectorXd::Zero(num_variables_);
  scene_->setJointPositions(q);

  // Create properly sized constraint
  double dt = 0.01;
  Eigen::VectorXd v_max = Eigen::VectorXd::Ones(num_variables_) * 1.0;
  auto vel_constraint = std::make_shared<VelocityLimit>(num_variables_, dt, v_max);

  // Create task
  auto target_pose =
      makeCartesianConfig(Eigen::Vector3d(0.3, 0.2, 0.5), Eigen::Quaterniond::Identity());
  auto task = std::make_shared<FrameTask>("tool0", target_pose);
  std::vector<std::shared_ptr<Task>> tasks = {task};
  std::vector<std::shared_ptr<Constraints>> constraints = {vel_constraint};

  Eigen::VectorXd delta_q;
  std::vector<std::shared_ptr<Barrier>> barriers;
  auto result = oink.solveIk(tasks, constraints, barriers, *scene_, delta_q);

  ASSERT_TRUE(result.has_value()) << "Solve failed: " << result.error();

  // After solve, verify workspace dimensions match what we expect
  EXPECT_EQ(oink.constraint_workspace_A.rows(), num_variables_);
  EXPECT_EQ(oink.constraint_workspace_A.cols(), num_variables_);
  EXPECT_EQ(oink.constraint_workspace_lower.rows(), num_variables_);
  EXPECT_EQ(oink.constraint_workspace_lower.cols(), 1);
  EXPECT_EQ(oink.constraint_workspace_upper.rows(), num_variables_);
  EXPECT_EQ(oink.constraint_workspace_upper.cols(), 1);
}

}  // namespace roboplan

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
