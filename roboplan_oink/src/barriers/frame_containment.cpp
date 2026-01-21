#include <roboplan_oink/barriers/frame_containment.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace roboplan {

FrameContainmentBarrier::FrameContainmentBarrier(const std::string& frame_name,
                                                 const Eigen::Vector3d& box_center,
                                                 const Eigen::Vector3d& box_dimensions,
                                                 int num_variables, double dt, double gain)
    : Barrier(6, dt, gain),  // 6 constraints: 2 per axis (upper + lower bounds)
      frame_name_(frame_name), box_center_(box_center), box_half_extents_(box_dimensions / 2.0),
      frame_position_(Eigen::Vector3d::Zero()),
      frame_jacobian_(Eigen::MatrixXd::Zero(6, num_variables)),
      position_jacobian_(Eigen::MatrixXd::Zero(3, num_variables)) {
  // Initialize barrier Jacobian workspace (6 × num_variables)
  jacobian_.resize(6, num_variables);
  jacobian_.setZero();
}

tl::expected<void, std::string> FrameContainmentBarrier::computeBarrier(const Scene& scene,
                                                                        Eigen::VectorXd& h) const {
  // Validate output size
  if (h.size() != 6) {
    return tl::make_unexpected("Barrier output vector size mismatch. Expected 6, got " +
                               std::to_string(h.size()));
  }

  // Get frame ID
  const auto maybe_frame_id = scene.getFrameId(frame_name_);
  if (!maybe_frame_id) {
    return tl::make_unexpected("Frame '" + frame_name_ + "' not found: " + maybe_frame_id.error());
  }
  const auto frame_id = maybe_frame_id.value();

  // Get current joint configuration
  const Eigen::VectorXd& q = scene.getCurrentJointPositions();

  // Get the model and data from scene
  const auto& model = scene.getModel();
  auto& data = scene.getData();

  // Update kinematics to ensure frame placements are current
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  // Get frame position from updated data
  const pinocchio::SE3& frame_transform = data.oMf.at(frame_id);
  frame_position_ = frame_transform.translation();

  // Compute barrier values for each axis
  // Upper bounds (h[0:3]): distance to upper face = (center + extent) - position
  h.head<3>() = (box_center_ + box_half_extents_) - frame_position_;

  // Lower bounds (h[3:6]): distance to lower face = position - (center - extent)
  h.tail<3>() = frame_position_ - (box_center_ - box_half_extents_);

  return {};
}

tl::expected<void, std::string>
FrameContainmentBarrier::computeJacobian(const Scene& scene, Eigen::MatrixXd& jacobian) const {
  // Validate output size
  if (jacobian.rows() != 6) {
    return tl::make_unexpected("Barrier Jacobian row size mismatch. Expected 6, got " +
                               std::to_string(jacobian.rows()));
  }

  // Get the frame ID
  const auto maybe_frame_id = scene.getFrameId(frame_name_);
  if (!maybe_frame_id) {
    return tl::make_unexpected("Frame '" + frame_name_ + "' not found: " + maybe_frame_id.error());
  }
  const auto frame_id = maybe_frame_id.value();

  // Get the model and data from scene
  const auto& model = scene.getModel();
  auto& data = scene.getData();

  // Get current joint configuration
  const Eigen::VectorXd& q = scene.getCurrentJointPositions();

  // Update kinematics to ensure everything is current
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  // Compute frame Jacobian (6 × num_variables) in world frame
  pinocchio::computeFrameJacobian(model, data, q, frame_id, pinocchio::ReferenceFrame::WORLD,
                                  frame_jacobian_);

  // Extract position part (first 3 rows)
  position_jacobian_ = frame_jacobian_.topRows<3>();

  // Form barrier Jacobian:
  // - Upper bound derivatives: ∂h_upper/∂q = ∂(center + extent - pos)/∂q = -∂pos/∂q
  jacobian.topRows<3>() = -position_jacobian_;

  // - Lower bound derivatives: ∂h_lower/∂q = ∂(pos - center + extent)/∂q = +∂pos/∂q
  jacobian.bottomRows<3>() = position_jacobian_;

  return {};
}

}  // namespace roboplan
