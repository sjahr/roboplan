#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <roboplan/core/scene.hpp>
#include <roboplan_oink/barriers/frame_containment.hpp>
#include <roboplan_oink/constraints/position_limit.hpp>
#include <roboplan_oink/constraints/velocity_limit.hpp>
#include <roboplan_oink/optimal_ik.hpp>
#include <roboplan_oink/tasks/frame.hpp>

#include <modules/optimal_ik.hpp>

namespace roboplan {

using namespace nanobind::literals;

void init_optimal_ik(nanobind::module_& m) {

  // Bind the abstract Task base class with shared_ptr holder
  nanobind::class_<Task>(m, "Task")
      .def_ro("gain", &Task::gain)
      .def_ro("weight", &Task::weight)
      .def_ro("lm_damping", &Task::lm_damping);

  // Bind FrameTaskParams configuration struct
  nanobind::class_<FrameTaskParams>(m, "FrameTaskParams")
      .def(nanobind::init<>())
      .def(
          "__init__",
          [](FrameTaskParams* self, double task_weight, double position_cost,
             double orientation_cost, double task_gain, double lm_damping) {
            new (self) FrameTaskParams{task_weight, position_cost, orientation_cost, task_gain,
                                       lm_damping};
          },
          "task_weight"_a = 1.0, "position_cost"_a = 1.0, "orientation_cost"_a = 1.0,
          "task_gain"_a = 1.0, "lm_damping"_a = 0.0)
      .def_rw("task_weight", &FrameTaskParams::task_weight)
      .def_rw("position_cost", &FrameTaskParams::position_cost)
      .def_rw("orientation_cost", &FrameTaskParams::orientation_cost)
      .def_rw("task_gain", &FrameTaskParams::task_gain)
      .def_rw("lm_damping", &FrameTaskParams::lm_damping);

  // Bind FrameTask inheriting from Task
  nanobind::class_<FrameTask, Task>(m, "FrameTask")
      .def(nanobind::init<const std::string&, const CartesianConfiguration&,
                          const FrameTaskParams&>(),
           "frame_name"_a, "target_pose"_a, "params"_a = FrameTaskParams{})
      .def_rw("frame_name", &FrameTask::frame_name)
      .def_rw("target_pose", &FrameTask::target_pose)
      .def("computeError", &FrameTask::computeError)
      .def("computeJacobian", &FrameTask::computeJacobian);

  // Bind the abstract Constraints base class
  nanobind::class_<Constraints>(m, "Constraints");

  // Bind PositionLimit constraint
  nanobind::class_<PositionLimit, Constraints>(m, "PositionLimit")
      .def(nanobind::init<int, double>(), "num_variables"_a, "gain"_a = 1.0)
      .def_rw("config_limit_gain", &PositionLimit::config_limit_gain)
      .def("computeQpConstraints", &PositionLimit::computeQpConstraints);

  // Bind VelocityLimit constraint
  nanobind::class_<VelocityLimit, Constraints>(m, "VelocityLimit")
      .def(nanobind::init<int, double, const Eigen::VectorXd&>(), "num_variables"_a, "dt"_a,
           "v_max"_a)
      .def_rw("dt", &VelocityLimit::dt)
      .def_rw("v_max", &VelocityLimit::v_max)
      .def("computeQpConstraints", &VelocityLimit::computeQpConstraints);

  // Bind the abstract Barrier base class
  nanobind::class_<Barrier>(m, "Barrier")
      .def_ro("barrier_dim", &Barrier::barrier_dim_)
      .def_rw("dt", &Barrier::dt_)
      .def_rw("gain", &Barrier::gain_);

  // Bind FrameContainmentBarrier
  nanobind::class_<FrameContainmentBarrier, Barrier>(m, "FrameContainmentBarrier")
      .def(nanobind::init<const std::string&, const Eigen::Vector3d&, const Eigen::Vector3d&, int,
                          double, double>(),
           "frame_name"_a, "box_center"_a, "box_dimensions"_a, "num_variables"_a, "dt"_a,
           "gain"_a = 1.0)
      .def_rw("frame_name", &FrameContainmentBarrier::frame_name_)
      .def_rw("box_center", &FrameContainmentBarrier::box_center_)
      .def_rw("box_half_extents", &FrameContainmentBarrier::box_half_extents_);

  // Bind Oink solver
  nanobind::class_<Oink>(m, "Oink")
      .def(nanobind::init<int>(), "num_variables"_a)
      .def(
          "solveIk",
          [](Oink& self, const std::vector<std::shared_ptr<Task>>& tasks,
             const std::vector<std::shared_ptr<Constraints>>& constraints,
             const std::vector<std::shared_ptr<Barrier>>& barriers,
             const std::shared_ptr<Scene>& scene) -> Eigen::VectorXd {
            Eigen::VectorXd delta_q;
            auto result = self.solveIk(tasks, constraints, barriers, *scene, delta_q);
            if (!result.has_value()) {
              throw std::runtime_error("IK solve failed: " + result.error());
            }
            return delta_q;
          },
          "tasks"_a, "constraints"_a, "barriers"_a, "scene"_a,
          "Solve inverse kinematics with constraints and barriers, return delta_q. Raises "
          "RuntimeError on failure.");
}

}  // namespace roboplan
