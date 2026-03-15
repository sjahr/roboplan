#pragma once

#include <Eigen/Dense>

#include <roboplan/core/scene.hpp>
#include <roboplan_oink/optimal_ik.hpp>

namespace roboplan {

/// @brief JointVelocityTask configuration.
struct JointVelocityTaskOptions {
  /// @brief Proportional gain for error feedback (default: 1.0).
  double task_gain = 1.0;

  /// @brief Levenberg-Marquardt damping for regularization (default: 0.0).
  double lm_damping = 0.0;
};

/// @brief Task for tracking a target joint velocity (displacement).
///
/// This task minimizes ||δq - δq_target||² where δq_target is the desired
/// joint velocity/displacement. This is useful for safety filtering, where
/// a nominal joint command is minimally modified to satisfy constraints.
///
/// Unlike ConfigurationTask which tracks a target configuration, this task
/// directly tracks a target velocity in the tangent space without needing
/// to integrate/differentiate between positions and velocities.
///
/// The task owns pre-allocated storage for its nv×nv Jacobian and nv error vector,
/// allocated at construction time to avoid runtime allocations during IK solving.
///
struct JointVelocityTask : public Task {
  /// @brief Target joint velocity/displacement to track.
  Eigen::VectorXd target_delta_q;

  /// @brief Per-joint weights for cost function.
  Eigen::VectorXd joint_weights;

  /// @brief Constructs a JointVelocityTask for tracking a target velocity.
  ///
  /// Pre-allocates storage for the nv×nv Jacobian and nv error vector based on
  /// joint_weights.size() (which equals model.nv).
  ///
  /// @param target_delta_q The target joint velocity/displacement.
  /// @param joint_weights Per-joint weights. All weights must be non-negative.
  /// @param options Optional task options.
  /// @throws std::invalid_argument if target_delta_q and joint_weights have different sizes
  ///         or if any joint weight is negative.
  JointVelocityTask(const Eigen::VectorXd& target_delta_q, const Eigen::VectorXd& joint_weights,
                    const JointVelocityTaskOptions& options = {});

  /// @brief Computes the velocity error.
  ///
  /// The error is simply target_delta_q, which when used in the QP formulation
  /// (minimize ||J*dq + gain*e||²) with J=-I yields:
  ///   ||(-I)*dq + gain*target||² = ||-dq + gain*target||²
  /// This is minimized when dq = gain*target_delta_q.
  ///
  /// @param scene The scene containing the robot model (used for dimension validation).
  /// @return Void if successful, else an error message string.
  tl::expected<void, std::string> computeError(const Scene& scene) override;

  /// @brief Computes the task Jacobian for the velocity task.
  ///
  /// The task Jacobian J(q) ∈ ℝ^(nv × nv) is the negative identity matrix (-I).
  /// Combined with error = target_delta_q, this yields:
  ///   minimize ||(-I)*dq + gain*target_delta_q||²
  ///          = ||-dq + gain*target_delta_q||²
  ///
  /// This is minimized at dq = gain*target_delta_q.
  ///
  /// Results are stored in jacobian_container.
  ///
  /// @param scene The scene containing the robot model (used for dimension validation).
  /// @return Void if successful, else an error message string.
  tl::expected<void, std::string> computeJacobian(const Scene& scene) override;

  /// @brief Creates a diagonal weight matrix from per-joint weights.
  ///
  /// The weight matrix W ∈ ℝ^(nv × nv) is constructed as:
  ///     W = diag(√joint_weights[i])
  ///
  /// @param joint_weights Per-joint weight vector.
  /// @return A nv×nv diagonal weight matrix.
  static Eigen::MatrixXd createWeightMatrix(const Eigen::VectorXd& joint_weights);
};

}  // namespace roboplan
