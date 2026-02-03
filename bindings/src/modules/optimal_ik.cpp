#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <roboplan/core/scene.hpp>
#include <roboplan_oink/barriers/position_barrier.hpp>
#include <roboplan_oink/constraints/position_limit.hpp>
#include <roboplan_oink/constraints/velocity_limit.hpp>
#include <roboplan_oink/optimal_ik.hpp>
#include <roboplan_oink/tasks/configuration.hpp>
#include <roboplan_oink/tasks/frame.hpp>

#include <modules/optimal_ik.hpp>

namespace roboplan {

using namespace nanobind::literals;

void init_optimal_ik(nanobind::module_& m) {

  nanobind::class_<Task>(m, "Task", "Abstract base class for IK tasks.")
      .def_ro("gain", &Task::gain, "Task gain for low-pass filtering.")
      .def_ro("weight", &Task::weight, "Weight matrix for cost normalization.")
      .def_ro("lm_damping", &Task::lm_damping, "Levenberg-Marquardt damping.")
      .def_ro("num_variables", &Task::num_variables, "Number of optimization variables.");

  // Bind FrameTaskOptions configuration struct
  nanobind::class_<FrameTaskOptions>(m, "FrameTaskOptions", "Parameters for FrameTask.")
      .def(nanobind::init<double, double, double, double>(), "position_cost"_a = 1.0,
           "orientation_cost"_a = 1.0, "task_gain"_a = 1.0, "lm_damping"_a = 0.0,
           "Constructor with custom parameters.")
      .def_rw("position_cost", &FrameTaskOptions::position_cost, "Position cost weight.")
      .def_rw("orientation_cost", &FrameTaskOptions::orientation_cost, "Orientation cost weight.")
      .def_rw("task_gain", &FrameTaskOptions::task_gain, "Task gain for low-pass filtering.")
      .def_rw("lm_damping", &FrameTaskOptions::lm_damping, "Levenberg-Marquardt damping.");

  // Bind FrameTask inheriting from Task
  nanobind::class_<FrameTask, Task>(m, "FrameTask",
                                    "Task to reach a target pose for a specified frame.")
      .def(nanobind::init<const std::string&, const CartesianConfiguration&, int,
                          const FrameTaskOptions&>(),
           "frame_name"_a, "target_pose"_a, "num_variables"_a, "options"_a = FrameTaskOptions{})
      .def_rw("frame_name", &FrameTask::frame_name, "Name of the frame to control.")
      .def_rw("target_pose", &FrameTask::target_pose, "Target pose for the frame.");

  // Bind ConfigurationTaskOptions configuration struct
  nanobind::class_<ConfigurationTaskOptions>(m, "ConfigurationTaskOptions",
                                             "Parameters for ConfigurationTask.")
      .def(nanobind::init<double, double>(), "task_gain"_a = 1.0, "lm_damping"_a = 0.0)
      .def_rw("task_gain", &ConfigurationTaskOptions::task_gain,
              "Task gain for low-pass filtering.")
      .def_rw("lm_damping", &ConfigurationTaskOptions::lm_damping, "Levenberg-Marquardt damping.");

  // Bind ConfigurationTask inheriting from Task
  nanobind::class_<ConfigurationTask, Task>(m, "ConfigurationTask",
                                            "Task to reach a target joint configuration.")
      .def(nanobind::init<const Eigen::VectorXd&, const Eigen::VectorXd&,
                          const ConfigurationTaskOptions&>(),
           "target_q"_a, "joint_weights"_a, "options"_a = ConfigurationTaskOptions{})
      .def_rw("target_q", &ConfigurationTask::target_q, "Target joint configuration.")
      .def_rw("joint_weights", &ConfigurationTask::joint_weights,
              "Weights for each joint in the configuration task.");

  // Bind the abstract Constraints base class
  nanobind::class_<Constraints>(m, "Constraints", "Abstract base class for IK constraints.");

  // Bind PositionLimit constraint
  nanobind::class_<PositionLimit, Constraints>(m, "PositionLimit",
                                               "Constraint to enforce joint position limits.")
      .def(nanobind::init<int, double>(), "num_variables"_a, "gain"_a = 1.0)
      .def_rw("config_limit_gain", &PositionLimit::config_limit_gain,
              "Gain for position limit enforcement.");

  // Bind VelocityLimit constraint
  nanobind::class_<VelocityLimit, Constraints>(m, "VelocityLimit",
                                               "Constraint to enforce joint velocity limits.")
      .def(nanobind::init<int, double, const Eigen::VectorXd&>(), "num_variables"_a, "dt"_a,
           "v_max"_a)
      .def_rw("dt", &VelocityLimit::dt, "Time step for velocity calculation.")
      .def_rw("v_max", &VelocityLimit::v_max, "Maximum joint velocities.");

  // Bind the abstract Barrier base class
  nanobind::class_<Barrier>(m, "Barrier", "Abstract base class for Control Barrier Functions.")
      .def("get_num_barriers", &Barrier::getNumBarriers, "scene"_a,
           "Get the number of barrier constraints.")
      .def_ro("gain", &Barrier::gain, "Barrier gain (gamma).")
      .def_ro("dt", &Barrier::dt, "Timestep.");

  // Bind PositionBarrier
  nanobind::class_<PositionBarrier, Barrier>(
      m, "PositionBarrier",
      "Position barrier constraint that keeps a frame within an axis-aligned bounding box.")
      // Full box constructor (all 3 axes)
      .def(nanobind::init<const std::string&, const Eigen::Vector3d&, const Eigen::Vector3d&, int,
                          double, double>(),
           "frame_name"_a, "p_min"_a, "p_max"_a, "num_variables"_a, "gain"_a = 1.0, "dt"_a = 0.01,
           "Create a position barrier for all 3 axes (x, y, z).")
      // Selective axis constructor
      .def(nanobind::init<const std::string&, const std::vector<int>&, const Eigen::VectorXd&,
                          const Eigen::VectorXd&, int, double, double>(),
           "frame_name"_a, "indices"_a, "p_min"_a, "p_max"_a, "num_variables"_a, "gain"_a = 1.0,
           "dt"_a = 0.01, "Create a position barrier for selected axes only.")
      .def("get_frame_position", &PositionBarrier::getFramePosition, "scene"_a,
           "Get the current frame position in world coordinates.")
      .def_ro("frame_name", &PositionBarrier::frame_name, "Name of the constrained frame.")
      .def_ro("indices", &PositionBarrier::indices, "Constrained axis indices.")
      .def_ro("p_min", &PositionBarrier::p_min, "Minimum position bounds.")
      .def_ro("p_max", &PositionBarrier::p_max, "Maximum position bounds.");

  // Bind Oink solver
  nanobind::class_<Oink>(m, "Oink", "Optimal Inverse Kinematics solver.")
      .def(nanobind::init<int>(), "num_variables"_a)
      .def(
          "solveIk",
          [](Oink& self, const std::vector<std::shared_ptr<Task>>& tasks,
             const std::vector<std::shared_ptr<Constraints>>& constraints,
             const std::vector<std::shared_ptr<Barrier>>& barriers,
             const std::shared_ptr<Scene>& scene, nanobind::DRef<Eigen::VectorXd> delta_q) {
            auto result = self.solveIk(tasks, constraints, barriers, *scene, delta_q);
            if (!result.has_value()) {
              throw std::runtime_error("IK solve failed: " + result.error());
            }
          },
          "tasks"_a, "constraints"_a, "barriers"_a, "scene"_a, "delta_q"_a,
          "Solve inverse kinematics for given tasks, constraints, and barriers.\n\n"
          "Solves a QP optimization problem to compute the joint velocity that minimizes\n"
          "weighted task errors while satisfying all constraints and barrier functions.\n"
          "The result is written directly into the provided delta_q buffer.\n\n"
          "Args:\n"
          "    tasks: List of weighted tasks to optimize for.\n"
          "    constraints: List of constraints to satisfy.\n"
          "    barriers: List of barrier functions for safety constraints.\n"
          "    scene: Scene containing robot model and state.\n"
          "    delta_q: Pre-allocated numpy array for output (size = num_variables).\n"
          "             Must be a contiguous float64 array. Modified in-place.\n\n"
          "Raises:\n"
          "    RuntimeError: If the QP solver fails to find a solution.\n\n"
          "Example:\n"
          "    delta_q = np.zeros(oink.num_variables)\n"
          "    oink.solveIk(tasks, constraints, barriers, scene, delta_q)");
}

}  // namespace roboplan
