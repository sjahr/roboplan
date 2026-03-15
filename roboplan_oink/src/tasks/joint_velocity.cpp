#include <roboplan_oink/tasks/joint_velocity.hpp>

namespace roboplan {

JointVelocityTask::JointVelocityTask(const Eigen::VectorXd& target_delta_q,
                                     const Eigen::VectorXd& joint_weights,
                                     const JointVelocityTaskOptions& options)
    : Task(createWeightMatrix(joint_weights), options.task_gain, options.lm_damping),
      target_delta_q(target_delta_q), joint_weights(joint_weights) {
  // Validate dimensions match
  if (target_delta_q.size() != joint_weights.size()) {
    throw std::invalid_argument("JointVelocityTask: target_delta_q size (" +
                                std::to_string(target_delta_q.size()) +
                                ") does not match joint_weights size (" +
                                std::to_string(joint_weights.size()) + ")");
  }

  // Validate joint weights are non-negative
  for (int i = 0; i < joint_weights.size(); ++i) {
    if (joint_weights(i) < 0.0) {
      throw std::invalid_argument("JointVelocityTask: joint_weights[" + std::to_string(i) +
                                  "] must be non-negative, got " +
                                  std::to_string(joint_weights(i)));
    }
  }

  // Pre-allocate storage: nv×nv Jacobian, nv error
  const int nv = static_cast<int>(joint_weights.size());
  initializeStorage(nv, nv);
}

tl::expected<void, std::string> JointVelocityTask::computeError(const Scene& scene) {
  const auto& model = scene.getModel();

  // Validate dimensions
  if (target_delta_q.size() != model.nv) {
    return tl::make_unexpected("JointVelocityTask: target_delta_q size (" +
                               std::to_string(target_delta_q.size()) + ") does not match model.nv (" +
                               std::to_string(model.nv) + ")");
  }

  // The error is target_delta_q so that when the QP minimizes ||J*dq + gain*e||²
  // with J = -I, we get: ||(-I)*dq + gain*target_delta_q||² = ||-dq + gain*target_delta_q||²
  // This is minimized when dq = gain*target_delta_q, which is exactly what we want.
  error_container = target_delta_q;

  return {};
}

tl::expected<void, std::string> JointVelocityTask::computeJacobian(const Scene& scene) {
  const auto& model = scene.getModel();

  // Validate joint_weights dimension against model
  if (joint_weights.size() != model.nv) {
    return tl::make_unexpected("JointVelocityTask: joint_weights size (" +
                               std::to_string(joint_weights.size()) +
                               ") does not match model.nv (" + std::to_string(model.nv) + ")");
  }

  // The Jacobian for velocity tracking is negative identity (-I) (nv × nv)
  jacobian_container.setIdentity();
  jacobian_container *= -1.0;

  return {};
}

Eigen::MatrixXd JointVelocityTask::createWeightMatrix(const Eigen::VectorXd& joint_weights) {
  const int nv = static_cast<int>(joint_weights.size());
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nv, nv);

  for (int i = 0; i < nv; ++i) {
    W(i, i) = std::sqrt(joint_weights(i));
  }

  return W;
}

}  // namespace roboplan
