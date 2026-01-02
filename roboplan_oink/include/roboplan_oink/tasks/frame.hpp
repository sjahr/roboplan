#pragma once

#include <memory>
#include <string>

#include <Eigen/Dense>

#include <roboplan/core/scene.hpp>
#include <roboplan/core/types.hpp>
#include <roboplan_oink/optimal_ik.hpp>

namespace roboplan {

/// @brief Optional parameters for FrameTask configuration.
struct FrameTaskParams {
  /// @brief Overall task weight for prioritization (default: 1.0).
  double task_weight = 1.0;

  /// @brief Cost weight for position error (default: 1.0).
  double position_cost = 1.0;

  /// @brief Cost weight for orientation error (default: 1.0).
  double orientation_cost = 1.0;

  /// @brief Proportional gain for error feedback (default: 1.0).
  double task_gain = 1.0;

  /// @brief Levenberg-Marquardt damping for regularization (default: 0.0).
  double lm_damping = 0.0;
};

/// @brief Task for tracking a target Cartesian pose with a specified frame.
///
/// This task computes the SE(3) error between a target pose and the current
/// frame pose, enabling full 6-DOF (position + orientation) tracking.
struct FrameTask : public Task {
  /// @brief Name of the frame to track (e.g., end-effector link name).
  std::string frame_name;

  /// @brief Target Cartesian configuration to reach.
  CartesianConfiguration target_pose;

  /// @brief Constructs a FrameTask for tracking a target pose.
  /// @param name The name of the frame to track.
  /// @param target_pose The target Cartesian configuration to reach.
  /// @param params Optional task parameters (default: all parameters set to defaults).
  FrameTask(const std::string& name, const CartesianConfiguration& target_pose,
            const FrameTaskParams& params = {})
      : Task(createWeightMatrix(params.task_weight, params.position_cost, params.orientation_cost),
             params.task_gain, params.lm_damping),
        frame_name(name), target_pose(target_pose) {}

  /// @brief Computes the SE(3) error between target and current frame pose.
  ///
  /// The error is computed as the logarithm of the relative transform:
  ///     error = log_6(T_frame_to_world^{-1} * T_target_to_world)
  ///
  /// @param scene The scene containing the robot model and current state.
  /// @param error Output vector (6D) containing the computed error.
  /// @return Void if successful, else an error message string.
  tl::expected<void, std::string> computeError(const Scene& scene,
                                               Eigen::VectorXd& error) const override;

  /// @brief Computes the task Jacobian for the frame tracking task.
  ///
  /// The task Jacobian J(q) ∈ ℝ^(6 × n_v) is the derivative of the task
  /// error e(q) ∈ ℝ^6 with respect to the configuration q. The formula is:
  ///
  ///     J(q) = -Jlog_6(T_target_to_frame) * J_frame(q)
  ///
  /// Where:
  /// - T_target_to_frame: Transform from target to current frame
  /// - J_frame(q): Frame Jacobian (expressed in frame coordinates)
  /// - Jlog_6: Pinocchio's logarithmic Jacobian
  ///
  /// @param scene The scene containing the robot model and current state.
  /// @param jacobian Output matrix (6 × nv) containing the computed Jacobian.
  /// @return Void if successful, else an error message string.
  tl::expected<void, std::string> computeJacobian(const Scene& scene,
                                                  Eigen::MatrixXd& jacobian) const override;

  /// @brief Creates a diagonal weight matrix from scalar cost weights.
  ///
  /// The weight matrix W ∈ ℝ^(6 × 6) is constructed as:
  ///     W = diag(√(task_weight * position_cost) * I_3,
  ///              √(task_weight * orientation_cost) * I_3)
  ///
  /// @param task_weight Overall task weight for prioritization.
  /// @param position_cost Cost weight for position error (first 3 dimensions).
  /// @param orientation_cost Cost weight for orientation error (last 3 dimensions).
  /// @return A 6×6 diagonal weight matrix.
  static Eigen::MatrixXd createWeightMatrix(double task_weight, double position_cost,
                                            double orientation_cost);

  // Pre-allocated logarithmic Jacobian
  Eigen::Matrix<double, 6, 6> Jlog = Eigen::Matrix<double, 6, 6>::Identity();
};

}  // namespace roboplan
