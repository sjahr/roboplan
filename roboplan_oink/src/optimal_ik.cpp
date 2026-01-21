#include <OsqpEigen/OsqpEigen.h>
#include <roboplan_oink/optimal_ik.hpp>

namespace roboplan {

namespace {
// Small regularization value added to Hessian diagonal for numerical stability
constexpr double kHessianRegularization = 1e-12;
}  // namespace

tl::expected<void, std::string> Task::computeQpObjective(const Eigen::MatrixXd& jacobian,
                                                         const Eigen::VectorXd& error,
                                                         Eigen::SparseMatrix<double>& H,
                                                         Eigen::VectorXd& c) const {
  const Eigen::VectorXd minus_gain_error = -gain * error;
  const Eigen::MatrixXd weighted_jacobian = weight * jacobian;
  const Eigen::VectorXd weighted_error = weight * minus_gain_error;

  const double mu = lm_damping * weighted_error.squaredNorm();

  Eigen::MatrixXd H_dense = weighted_jacobian.transpose() * weighted_jacobian;
  H_dense.diagonal().array() += mu;
  H = H_dense.sparseView();

  c = (-weighted_error.transpose() * weighted_jacobian).transpose();

  return {};
}

Barrier::Barrier(int barrier_dim, double dt, double gain)
    : barrier_dim_(barrier_dim), dt_(dt), gain_(gain), h_(Eigen::VectorXd::Zero(barrier_dim)) {
  // Note: jacobian_ is not initialized here with num_variables since it's not known yet.
  // Derived classes must set the size in their constructor.
}

tl::expected<void, std::string>
Barrier::computeQpConstraint(const Scene& scene, Eigen::Ref<Eigen::MatrixXd> constraint_matrix,
                             Eigen::Ref<Eigen::VectorXd> lower_bounds,
                             Eigen::Ref<Eigen::VectorXd> upper_bounds) {
  // Validate workspace dimensions
  if (constraint_matrix.rows() != barrier_dim_) {
    return tl::make_unexpected("Barrier constraint matrix row dimension mismatch. Expected " +
                               std::to_string(barrier_dim_) + ", got " +
                               std::to_string(constraint_matrix.rows()));
  }

  // Step 1: Compute barrier values h(q)
  auto barrier_result = computeBarrier(scene, h_);
  if (!barrier_result.has_value()) {
    return tl::make_unexpected("Failed to compute barrier function: " + barrier_result.error());
  }

  // Step 2: Compute barrier Jacobian ∂h/∂q
  auto jacobian_result = computeJacobian(scene, jacobian_);
  if (!jacobian_result.has_value()) {
    return tl::make_unexpected("Failed to compute barrier Jacobian: " + jacobian_result.error());
  }

  // Step 3: Form discrete-time barrier constraint
  // Continuous-time: ∂h/∂q · q̇ + gain·h(q) ≥ 0
  // Discrete-time: -J_h/dt · Δq ≤ gain·h(q)
  // In standard form: G·Δq ≤ h_bounds where G = -J_h/dt
  constraint_matrix = -jacobian_ / dt_;
  lower_bounds.setConstant(-OsqpEigen::INFTY);
  upper_bounds = gain_ * h_;

  return {};
}

Oink::Oink(int num_variables)
    : num_variables(num_variables), task_J(Eigen::MatrixXd::Zero(num_variables, num_variables)),
      task_e(Eigen::VectorXd::Zero(num_variables)), task_c(Eigen::VectorXd::Zero(num_variables)),
      task_H(num_variables, num_variables), H(num_variables, num_variables),
      c(Eigen::VectorXd::Zero(num_variables)) {
  settings.setWarmStart(true);
  settings.setVerbosity(false);
}

Oink::Oink(int num_variables, const OsqpEigen::Settings& custom_settings)
    : settings(custom_settings), num_variables(num_variables),
      task_J(Eigen::MatrixXd::Zero(num_variables, num_variables)),
      task_e(Eigen::VectorXd::Zero(num_variables)), task_c(Eigen::VectorXd::Zero(num_variables)),
      task_H(num_variables, num_variables), H(num_variables, num_variables),
      c(Eigen::VectorXd::Zero(num_variables)) {}

tl::expected<void, std::string>
Oink::solveIk(const std::vector<std::shared_ptr<Task>>& tasks,
              const std::vector<std::shared_ptr<Constraints>>& constraints,
              const std::vector<std::shared_ptr<Barrier>>& barriers, const Scene& scene,
              Eigen::VectorXd& delta_q) {
  // Reset Hessian and Gradient
  H.setIdentity();
  H.diagonal().array() *= kHessianRegularization;
  c.setZero();

  // Calculate accumulated Hessian and Gradient
  for (const auto& task : tasks) {
    auto jacobian_result = task->computeJacobian(scene, task_J);
    if (!jacobian_result.has_value()) {
      return tl::make_unexpected("Failed to compute Jacobian: " + jacobian_result.error());
    }

    auto error_result = task->computeError(scene, task_e);
    if (!error_result.has_value()) {
      return tl::make_unexpected("Failed to compute error: " + error_result.error());
    }

    auto objective_result = task->computeQpObjective(task_J, task_e, task_H, task_c);
    if (!objective_result.has_value()) {
      return tl::make_unexpected("Failed to compute QP objective: " + objective_result.error());
    }

    H += task_H;
    c += task_c;
  }
  H.makeCompressed();

  // Query total constraint dimensions and cache sizes to avoid redundant calls
  constraint_sizes.reserve(constraints.size());
  int total_constraint_rows = 0;
  for (const auto& constraint : constraints) {
    int num_rows = constraint->getNumConstraints(scene);
    constraint_sizes.push_back(num_rows);
    total_constraint_rows += num_rows;
  }

  const bool init_required =
      !solver.isInitialized() || (total_constraint_rows != last_constraint_rows);

  // Resize constraint workspace only if dimensions changed (zero allocations in steady state)
  // Note: Workspace is allocated even when total_constraint_rows == 0 for consistency,
  // but OSQP constraint matrices are only set when constraints exist (for unconstrained QP)
  if (init_required) {
    constraint_workspace_A.resize(total_constraint_rows, num_variables);
    constraint_workspace_lower.resize(total_constraint_rows);
    constraint_workspace_upper.resize(total_constraint_rows);

    // Resize sparse workspace to match
    A_sparse.resize(total_constraint_rows, num_variables);

    last_constraint_rows = total_constraint_rows;
  }

  // Fill constraint matrices block by block using Eigen::Ref (zero-copy views)
  int row_offset = 0;
  for (size_t i = 0; i < constraints.size(); ++i) {
    const int num_rows = constraint_sizes.at(i);

    // Safety check: ensure we don't exceed workspace bounds
    if (row_offset + num_rows > total_constraint_rows) {
      return tl::make_unexpected("Internal error: constraint row offset " +
                                 std::to_string(row_offset + num_rows) + " exceeds total rows " +
                                 std::to_string(total_constraint_rows));
    }

    // Create Eigen::Ref views into the workspace (zero-copy, no allocation)
    Eigen::Ref<Eigen::MatrixXd> constraint_A_view =
        constraint_workspace_A.middleRows(row_offset, num_rows);
    Eigen::Ref<Eigen::VectorXd> constraint_lower_view =
        constraint_workspace_lower.segment(row_offset, num_rows);
    Eigen::Ref<Eigen::VectorXd> constraint_upper_view =
        constraint_workspace_upper.segment(row_offset, num_rows);

    // Compute constraints directly into workspace views
    auto constraint_result = constraints.at(i)->computeQpConstraints(
        scene, constraint_A_view, constraint_lower_view, constraint_upper_view);
    if (!constraint_result.has_value()) {
      return tl::make_unexpected("Failed to compute constraints: " + constraint_result.error());
    }

    // Validation: Ensure constraint didn't resize the views (safety check)
    if (constraint_A_view.rows() != num_rows || constraint_A_view.cols() != num_variables) {
      return tl::make_unexpected("Constraint implementation error: resized output matrices. "
                                 "Expected (" +
                                 std::to_string(num_rows) + " x " + std::to_string(num_variables) +
                                 "), got (" + std::to_string(constraint_A_view.rows()) + " x " +
                                 std::to_string(constraint_A_view.cols()) + ")");
    }

    row_offset += num_rows;
  }
  // Clear constraint_sizes for the next iteration
  constraint_sizes.clear();

  // === Barrier Processing ===
  // Query total barrier dimensions and cache sizes
  barrier_sizes.reserve(barriers.size());
  int total_barrier_rows = 0;
  for (const auto& barrier : barriers) {
    int num_rows = barrier->barrier_dim_;
    barrier_sizes.push_back(num_rows);
    total_barrier_rows += num_rows;
  }

  const bool barrier_init_required = (total_barrier_rows != last_barrier_rows);

  // Resize barrier workspace only if dimensions changed
  if (barrier_init_required) {
    barrier_workspace_A.resize(total_barrier_rows, num_variables);
    barrier_workspace_lower.resize(total_barrier_rows);
    barrier_workspace_upper.resize(total_barrier_rows);
    last_barrier_rows = total_barrier_rows;
  }

  // Fill barrier matrices block by block
  row_offset = 0;
  for (size_t i = 0; i < barriers.size(); ++i) {
    const int num_rows = barrier_sizes.at(i);

    if (row_offset + num_rows > total_barrier_rows) {
      return tl::make_unexpected("Internal error: barrier row offset " +
                                 std::to_string(row_offset + num_rows) + " exceeds total rows " +
                                 std::to_string(total_barrier_rows));
    }

    // Create Eigen::Ref views into the workspace
    Eigen::Ref<Eigen::MatrixXd> barrier_A_view =
        barrier_workspace_A.middleRows(row_offset, num_rows);
    Eigen::Ref<Eigen::VectorXd> barrier_lower_view =
        barrier_workspace_lower.segment(row_offset, num_rows);
    Eigen::Ref<Eigen::VectorXd> barrier_upper_view =
        barrier_workspace_upper.segment(row_offset, num_rows);

    // Compute barrier constraints directly into workspace views
    auto barrier_result = barriers.at(i)->computeQpConstraint(
        scene, barrier_A_view, barrier_lower_view, barrier_upper_view);
    if (!barrier_result.has_value()) {
      return tl::make_unexpected("Failed to compute barrier constraints: " +
                                 barrier_result.error());
    }

    row_offset += num_rows;
  }
  barrier_sizes.clear();

  // Combine constraints and barriers into a single constraint matrix
  const int total_combined_rows = total_constraint_rows + total_barrier_rows;
  const bool combined_init_required = init_required || barrier_init_required;

  Eigen::MatrixXd combined_A(total_combined_rows, num_variables);
  Eigen::VectorXd combined_lower(total_combined_rows);
  Eigen::VectorXd combined_upper(total_combined_rows);

  // Stack constraints first, then barriers
  if (total_constraint_rows > 0) {
    combined_A.topRows(total_constraint_rows) = constraint_workspace_A;
    combined_lower.head(total_constraint_rows) = constraint_workspace_lower;
    combined_upper.head(total_constraint_rows) = constraint_workspace_upper;
  }
  if (total_barrier_rows > 0) {
    combined_A.bottomRows(total_barrier_rows) = barrier_workspace_A;
    combined_lower.tail(total_barrier_rows) = barrier_workspace_lower;
    combined_upper.tail(total_barrier_rows) = barrier_workspace_upper;
  }

  // Convert combined constraint matrix to sparse format for OSQP
  A_sparse = combined_A.sparseView();

  if (combined_init_required) {
    // Clear previous solver state and data if it exists
    if (solver.isInitialized()) {
      solver.clearSolver();
    }
    // Clear previous data matrices to allow re-setting them
    solver.data()->clearHessianMatrix();
    solver.data()->clearLinearConstraintsMatrix();

    // Apply solver settings by copying from the stored settings
    const OSQPSettings* stored_settings = settings.getSettings();
    solver.settings()->setWarmStart(stored_settings->warm_starting);
    solver.settings()->setVerbosity(stored_settings->verbose);
    solver.settings()->setAlpha(stored_settings->alpha);
    solver.settings()->setAbsoluteTolerance(stored_settings->eps_abs);
    solver.settings()->setRelativeTolerance(stored_settings->eps_rel);
    solver.settings()->setPrimalInfeasibilityTolerance(stored_settings->eps_prim_inf);
    solver.settings()->setDualInfeasibilityTolerance(stored_settings->eps_dual_inf);
    solver.settings()->setMaxIteration(stored_settings->max_iter);
    solver.settings()->setRho(stored_settings->rho);
    solver.settings()->setPolish(stored_settings->polishing);
    solver.settings()->setAdaptiveRho(stored_settings->adaptive_rho);
    solver.settings()->setTimeLimit(stored_settings->time_limit);

    // Initialize solver with new dimensions
    solver.data()->setNumberOfVariables(num_variables);
    solver.data()->setNumberOfConstraints(total_combined_rows);
    if (total_combined_rows > 0) {
      solver.data()->setLinearConstraintsMatrix(A_sparse);
      solver.data()->setLowerBound(combined_lower);
      solver.data()->setUpperBound(combined_upper);
    }
    solver.data()->setHessianMatrix(H);
    solver.data()->setGradient(c);
    if (!solver.initSolver()) {
      return tl::make_unexpected("Failed to initialize solver");
    }
  } else {
    if (!solver.updateHessianMatrix(H)) {
      return tl::make_unexpected("Failed to update Hessian matrix");
    }

    if (!solver.updateGradient(c)) {
      return tl::make_unexpected("Failed to update gradient vector");
    }

    if (total_combined_rows > 0) {
      if (!solver.updateLinearConstraintsMatrix(A_sparse)) {
        return tl::make_unexpected("Failed to update linear constraints matrix");
      }
      if (!solver.updateBounds(combined_lower, combined_upper)) {
        return tl::make_unexpected("Failed to update constraint bounds");
      }
    }
  }

  // Solve the QP problem
  auto result = solver.solveProblem();
  if (result != OsqpEigen::ErrorExitFlag::NoError) {
    return tl::make_unexpected("QP solver failed to find a solution");
  }

  // Extract the solution and update delta_q
  delta_q = solver.getSolution();

  return {};
}

}  // namespace roboplan
