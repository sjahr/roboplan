#pragma once

#include <string>

#include <Eigen/Dense>

#include <roboplan/core/scene.hpp>
#include <roboplan_oink/optimal_ik.hpp>

namespace roboplan {

/// @brief Control barrier function for constraining a frame's position within a 3D box
///
/// This barrier enforces that a specified frame's position remains within a box-shaped region.
/// The box is defined by its center and dimensions (half-extents). The barrier generates 6
/// inequality constraints (2 per axis) to enforce the containment.
///
/// Barrier function:
/// - For axis i ∈ {x, y, z}:
///   - Upper bound: h_i = (center_i + extent_i) - position_i ≥ 0
///   - Lower bound: h_{i+3} = position_i - (center_i - extent_i) ≥ 0
///
/// Where position_i is the frame's position in world coordinates.
struct FrameContainmentBarrier : public Barrier {
  /// @brief Constructor
  ///
  /// @param frame_name Name of the frame to constrain (e.g., "ee_link")
  /// @param box_center Center of the containment box in world coordinates (x, y, z)
  /// @param box_dimensions Full dimensions of the box (width, height, depth)
  /// @param num_variables Number of optimization variables (DOF count)
  /// @param dt Time step for discretization
  /// @param gain Gain parameter for barrier strength (default: 1.0)
  FrameContainmentBarrier(const std::string& frame_name, const Eigen::Vector3d& box_center,
                          const Eigen::Vector3d& box_dimensions, int num_variables, double dt,
                          double gain = 1.0);

  /// @brief Compute barrier function values h(q) at current configuration
  ///
  /// Returns a 6D vector where:
  /// - h[0:3] = (box_center + box_half_extents) - frame_position (upper bounds)
  /// - h[3:6] = frame_position - (box_center - box_half_extents) (lower bounds)
  ///
  /// Positive values indicate the frame is safely inside the box.
  /// Negative values indicate constraint violation.
  ///
  /// @param scene The scene containing robot state and model
  /// @param h Output barrier values (must be pre-allocated to size 6)
  /// @return void on success, error message on failure
  tl::expected<void, std::string> computeBarrier(const Scene& scene,
                                                 Eigen::VectorXd& h) const override;

  /// @brief Compute barrier Jacobian ∂h/∂q at current configuration
  ///
  /// Returns a 6 × num_variables matrix where:
  /// - Rows 0:3 correspond to upper bound constraints (negative of position Jacobian)
  /// - Rows 3:6 correspond to lower bound constraints (positive position Jacobian)
  ///
  /// @param scene The scene containing robot state and model
  /// @param jacobian Output Jacobian matrix (must be pre-allocated to size 6 × num_variables)
  /// @return void on success, error message on failure
  tl::expected<void, std::string> computeJacobian(const Scene& scene,
                                                  Eigen::MatrixXd& jacobian) const override;

  // Frame and box parameters
  std::string frame_name_;            // Name of the frame to constrain
  Eigen::Vector3d box_center_;        // Center of the containment box (world coords)
  Eigen::Vector3d box_half_extents_;  // Half-dimensions of the box (dx/2, dy/2, dz/2)

  // Pre-allocated workspace
  mutable Eigen::Vector3d frame_position_;     // Current frame position (world coords)
  mutable Eigen::MatrixXd frame_jacobian_;     // Frame Jacobian (6 × num_variables)
  mutable Eigen::MatrixXd position_jacobian_;  // Position part only (3 × num_variables)
};

}  // namespace roboplan
