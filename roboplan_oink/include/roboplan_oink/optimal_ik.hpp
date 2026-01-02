#pragma once

#include <memory>
#include <string>

#include "OsqpEigen/OsqpEigen.h"
#include <tl/expected.hpp>

#include <roboplan/core/scene.hpp>
#include <roboplan/core/types.hpp>

namespace roboplan {
struct Task {
  Task(Eigen::MatrixXd weight_matrix, double task_gain = 1.0, double lm_damp = 0.0)
      : gain(task_gain), weight(weight_matrix), lm_damping(lm_damp) {}
  virtual ~Task() = default;

  virtual tl::expected<void, std::string> computeJacobian(const Scene& scene,
                                                          Eigen::MatrixXd& jacobian) const = 0;
  virtual tl::expected<void, std::string> computeError(const Scene& scene,
                                                       Eigen::VectorXd& error) const = 0;

  /// @brief Compute QP objective matrices (H, c) for this task.
  ///
  /// Computes the contribution of this task to the quadratic program objective:
  ///     minimize  ½ ‖J Δq + α e‖²_W
  ///
  /// This is equivalent to:
  ///     minimize  ½ Δq^T H Δq + c^T Δq
  ///
  /// Where:
  /// - J: Task Jacobian matrix
  /// - Δq: Configuration displacement
  /// - α: Task gain for low-pass filtering
  /// - e: Task error vector
  /// - W: Weight matrix for cost normalization
  ///
  /// The method returns:
  /// - H = J^T W^T W J + μ I  (num_variables x num_variables Hessian matrix, sparse)
  /// - c = -e^T W^T W J        (num_variables x 1 linear term)
  ///
  /// Where μ is the Levenberg-Marquardt damping term.
  /// @param jacobian Task Jacobian matrix
  /// @param error Task error vector
  /// @param H Hessian matrix
  /// @param c Linear cost term
  tl::expected<void, std::string> computeQpObjective(const Eigen::MatrixXd& jacobian,
                                                     const Eigen::VectorXd& error,
                                                     Eigen::SparseMatrix<double>& H,
                                                     Eigen::VectorXd& c) const;

  const double gain = 1.0;        // Task gain for low-pass filtering
  const Eigen::MatrixXd weight;   // Weight matrix for cost normalization
  const double lm_damping = 0.0;  // Levenberg-Marquardt damping
};

struct Constraints {
  virtual ~Constraints() = default;

  /// @brief Get the number of constraint rows this constraint will produce
  /// @param scene The scene containing robot state and model
  /// @return Number of constraint rows
  virtual int getNumConstraints(const Scene& scene) const = 0;

  /// @brief Compute QP constraint matrices using pre-allocated workspace views
  ///
  /// The constraint_matrix, lower_bounds, and upper_bounds parameters are Eigen::Ref views
  /// into pre-allocated workspace memory. The views are already sized to match
  /// getNumConstraints() rows, so implementations should fill the entire view.
  ///
  /// @param scene The scene containing robot state and model
  /// @param constraint_matrix Output constraint matrix G (pre-sized view: num_constraints ×
  /// num_variables)
  /// @param lower_bounds Output lower bounds vector (pre-sized view: num_constraints)
  /// @param upper_bounds Output upper bounds vector (pre-sized view: num_constraints)
  /// @return void on success, error message on failure
  virtual tl::expected<void, std::string>
  computeQpConstraints(const Scene& scene, Eigen::Ref<Eigen::MatrixXd> constraint_matrix,
                       Eigen::Ref<Eigen::VectorXd> lower_bounds,
                       Eigen::Ref<Eigen::VectorXd> upper_bounds) = 0;
};

/// @brief Oink - Optimal Inverse Kinematics solver
struct Oink {
  /// @brief Constructor that initializes matrices and solver with given dimensions
  ///
  /// @param num_variables Number of optimization variables (typically number of actuatable DOFs)
  Oink(int num_variables);

  /// @brief Constructor with custom settings
  ///
  /// @param num_variables Number of optimization variables (typically number of DOFs)
  /// @param custom_settings Custom OSQP solver settings
  Oink(int num_variables, const OsqpEigen::Settings& custom_settings);

  /// @brief Solve inverse kinematics for given tasks and constraints
  ///
  /// @param tasks Vector of weighted tasks to optimize for
  /// @param constraints Vector of constraints to satisfy
  /// @param scene Scene containing robot model and state
  /// @param delta_q Output configuration displacement
  /// @return void on success, error message on failure
  tl::expected<void, std::string>
  solveIk(const std::vector<std::shared_ptr<Task>>& tasks,
          const std::vector<std::shared_ptr<Constraints>>& constraints, const Scene& scene,
          Eigen::VectorXd& delta_q);

  // QP solver
  OsqpEigen::Solver solver;
  OsqpEigen::Settings settings;

  // Problem dimensions
  int num_variables;

  // Pre-allocated task matrices (reused for each task)
  Eigen::MatrixXd task_J;
  Eigen::VectorXd task_e;
  Eigen::VectorXd task_c;
  Eigen::SparseMatrix<double> task_H;

  // Pre-allocated accumulated QP matrices
  Eigen::SparseMatrix<double> H;
  Eigen::VectorXd c;

  // Pre-allocated constraint matrices
  Eigen::MatrixXd constraint_workspace_A;
  Eigen::VectorXd constraint_workspace_lower;
  Eigen::VectorXd constraint_workspace_upper;
  Eigen::SparseMatrix<double> A_sparse;
  std::vector<int> constraint_sizes;
  int last_constraint_rows = -1;  // -1 indicates uninitialized
};
}  // namespace roboplan
