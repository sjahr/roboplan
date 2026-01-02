#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include <pinocchio/algorithm/kinematics.hpp>
#include <roboplan/core/scene.hpp>
#include <roboplan_example_models/resources.hpp>
#include <roboplan_oink/constraints/position_limit.hpp>
#include <roboplan_oink/optimal_ik.hpp>
#include <roboplan_oink/tasks/frame.hpp>

using namespace roboplan;

int main(int /*argc*/, char* /*argv*/[]) {
  // Set up the scene
  const auto share_prefix = example_models::get_package_share_dir();
  const auto model_prefix = share_prefix / "roboplan_example_models" / "models";
  const auto urdf_path = model_prefix / "ur_robot_model" / "ur5_gripper.urdf";
  const auto srdf_path = model_prefix / "ur_robot_model" / "ur5_gripper.srdf";
  const std::vector<std::filesystem::path> package_paths = {share_prefix};

  auto scene = Scene("example_optimal_ik_scene", urdf_path, srdf_path, package_paths);

  // Set up target configuration
  const auto& model = scene.getModel();
  auto q_tgt = pinocchio::neutral(model);
  q_tgt[0] += 0.1;
  q_tgt[2] -= 0.1;
  q_tgt[4] -= 0.25;

  // Define goal pose
  const auto goal = CartesianConfiguration{
      .base_frame = "base",
      .tip_frame = "tool0",
      .tform = scene.forwardKinematics(q_tgt, "tool0"),
  };

  // Set initial configuration
  auto q_start = pinocchio::neutral(model);
  scene.setJointPositions(q_start);

  // Update forward kinematics in scene data
  pinocchio::forwardKinematics(model, scene.getData(), q_start);
  pinocchio::updateFramePlacements(model, scene.getData());

  // Create a FrameTask
  FrameTaskParams params{
      .task_weight = 1.0,
      .position_cost = 1.0,
      .orientation_cost = 1.0,
      .task_gain = 1.0,
      .lm_damping = 0.01,
  };
  auto frame_task = std::make_shared<FrameTask>("tool0", goal, params);

  // Add task to vector
  std::vector<std::shared_ptr<Task>> tasks = {frame_task};

  // Get number of variables for constraints and solver
  const int num_variables = model.nv;

  // Create position limit constraint
  auto position_limit = std::make_shared<PositionLimit>(num_variables, 1.0);  // gain = 1.0
  std::vector<std::shared_ptr<Constraints>> constraints = {position_limit};

  // Create Oink instance
  Oink oink(num_variables);

  // Solve IK with constraints
  Eigen::VectorXd delta_q;
  auto result = oink.solveIk(tasks, constraints, scene, delta_q);

  if (!result.has_value()) {
    std::cout << "IK solve failed: " << result.error() << "\n";
    return 1;
  }

  std::cout << "IK solve succeeded!\n";
  std::cout << "  Start config: " << q_start.transpose() << "\n";
  std::cout << "  Delta q:      " << delta_q.transpose() << "\n";
  std::cout << "  Final config: " << (q_start + delta_q).transpose() << "\n";
  std::cout << "  Target config:" << q_tgt.transpose() << "\n";

  return 0;
}
