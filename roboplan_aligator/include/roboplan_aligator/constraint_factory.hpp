#pragma once

#include <aligator/core/constraint-set.hpp>          // ConstraintSetTpl
#include <aligator/core/function-abstract.hpp>       // StageFunctionTpl
#include <aligator/third-party/polymorphic_cxx14.h>  // xyz::polymorphic

#include <roboplan_aligator/constraints/torque_limit.hpp>  // spec type

#include <roboplan_aligator/problem_builder.hpp>  // PhaseSpace

namespace roboplan {

class ReducedGroupModel;

namespace aligator_detail {

struct ConstraintPair {
  xyz::polymorphic<aligator::StageFunctionTpl<double>> func;
  xyz::polymorphic<aligator::ConstraintSetTpl<double>> set;
};

// `buildConstraint` is an overload set keyed on the spec type: each overload builds the
// (residual, set) pair for its spec. `is_terminal` is true when the constraint targets the
// terminal node; an overload whose residual needs a control (e.g. TorqueLimit) throws
// std::invalid_argument in that case, so the rejection lives next to the build rather than in the
// dispatcher.
//
// Every ConstraintSpec alternative must have exactly one overload here. The dispatcher in
// trajectory_optimizer.cpp is a type-blind std::visit over ConstraintSpec, so a missing overload
// is a compile error.

ConstraintPair buildConstraint(const PhaseSpace& space, const ReducedGroupModel& rgm,
                               const TorqueLimit& spec, bool is_terminal);

}  // namespace aligator_detail

}  // namespace roboplan
