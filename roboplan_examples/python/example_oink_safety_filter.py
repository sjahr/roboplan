#!/usr/bin/env python3
"""
Safety Filter Example using Oink.

This example demonstrates using the QP solver as a safety filter:
1. First, solve IK with FrameTask + constraints (no barriers) to get a nominal delta_q
2. Then, use JointVelocityTask to minimally modify delta_q while satisfying barriers

This architecture separates the IK computation from safety enforcement, which can be
useful when the nominal command comes from an external source (teleoperation, trajectory
playback, learned policy, etc.).
"""

import sys
import threading
import time
import tyro
import xacro

import numpy as np
import pinocchio as pin

from common import MODELS
from roboplan.core import Scene, CartesianConfiguration
from roboplan.example_models import get_package_share_dir
from roboplan.optimal_ik import (
    ConfigurationTask,
    ConfigurationTaskOptions,
    FrameTask,
    FrameTaskOptions,
    JointVelocityTask,
    JointVelocityTaskOptions,
    Oink,
    PositionBarrier,
    PositionLimit,
    VelocityLimit,
)
from roboplan.viser_visualizer import ViserVisualizer


def main(
    model: str = "ur5",
    task_gain: float = 1.0,
    lm_damping: float = 2.0,
    control_freq: float = 100.0,
    max_joint_velocity: float = 1.0,
    barrier_gain: float = 100.0,
    barrier_size: float = 1.0,
    safety_margin: float = 0.1,
    host: str = "localhost",
    port: str = "8000",
):
    """
    Run the safety filter example with two-stage IK solving.

    Stage 1: Solve IK with FrameTask + constraints (no barriers) -> nominal delta_q
    Stage 2: Filter delta_q with JointVelocityTask + constraints + barriers -> safe delta_q

    This demonstrates using the QP as a pure safety filter, which is useful when the
    nominal command comes from an external source.

    Parameters:
        model: The name of the model to use (ur5, franka, or dual).
        task_gain: Task gain (alpha) for the IK solver (0-1).
        lm_damping: Levenberg-Marquardt damping for regularization.
        control_freq: Control loop frequency in Hz.
        max_joint_velocity: Maximum joint velocity in rad/s for all joints.
        barrier_gain: Barrier gain for CBF constraint (higher = more conservative).
        barrier_size: Size of the cubic barrier box around the EE start position (meters).
        safety_margin: Distance from boundary where barrier activates (meters).
        host: The host for the ViserVisualizer.
        port: The port for the ViserVisualizer.
    """

    if model not in MODELS:
        print(f"Invalid model requested: {model}")
        sys.exit(1)

    model_data = MODELS[model]
    package_paths = [get_package_share_dir()]

    # Pre-process with xacro. This is not necessary for raw URDFs.
    urdf_xml = xacro.process_file(model_data.urdf_path).toxml()
    srdf_xml = xacro.process_file(model_data.srdf_path).toxml()

    # Specify argument names to distinguish overloaded Scene constructors from python.
    scene = Scene(
        "oink_scene",
        urdf=urdf_xml,
        srdf=srdf_xml,
        package_paths=package_paths,
        yaml_config_path=model_data.yaml_config_path,
    )

    # Print joint information
    print(f"\n=== Safety Filter Example: {model} ===")
    joint_names = scene.getJointNames()
    actuated_joint_names = scene.getActuatedJointNames()
    print(f"Total joints: {len(joint_names)}")
    print(f"Actuated joints: {len(actuated_joint_names)}")

    q_full = scene.getCurrentJointPositions()

    # Create a redundant Pinocchio model just for visualization.
    model_pin = pin.buildModelFromXML(urdf_xml)
    collision_model = pin.buildGeomFromUrdfString(
        model_pin, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
    )
    visual_model = pin.buildGeomFromUrdfString(
        model_pin, urdf_xml, pin.GeometryType.VISUAL, package_dirs=package_paths
    )

    viz = ViserVisualizer(model_pin, collision_model, visual_model)
    viz.initViewer(open=True, loadModel=True, host=host, port=port)

    # Determine the velocity space dimension (nv) by computing a Jacobian
    jac = scene.computeFrameJacobian(q_full, model_data.ee_names[0])
    num_variables = jac.shape[1]  # nv (velocity space dimension)
    print(f"\nConfiguration space dimension (nq): {len(q_full)}")
    print(f"Velocity space dimension (nv): {num_variables}")

    # Set up the Oink solver
    oink = Oink(num_variables)

    # Thread-safe access to scene
    scene_lock = threading.Lock()

    # Control loop time step
    dt = 1.0 / control_freq

    # Create position limit constraint (used in both stages)
    position_limit = PositionLimit(num_variables, gain=1.0)

    # Create velocity limit constraint (used in both stages)
    v_max = np.full(num_variables, max_joint_velocity)
    velocity_limit = VelocityLimit(num_variables, dt, v_max)

    # Constraints for Stage 1 (IK) and Stage 2 (Safety Filter)
    constraints = [position_limit, velocity_limit]

    # Get initial EE position for barrier creation
    initial_ee_pose = scene.forwardKinematics(q_full, model_data.ee_names[0])
    initial_ee_pos = initial_ee_pose[:3, 3]

    # Create position barrier (only used in Stage 2 - Safety Filter)
    half_size = barrier_size / 2.0
    p_min = initial_ee_pos - half_size
    p_max = initial_ee_pos + half_size
    position_barrier = PositionBarrier(
        model_data.ee_names[0],
        p_min,
        p_max,
        num_variables,
        gain=barrier_gain,
        dt=dt,
        safe_displacement_gain=1.0,
        safety_margin=safety_margin,
    )
    barriers = [position_barrier]

    print(f"\n=== Two-Stage Architecture ===")
    print(f"Stage 1 (IK): FrameTask + constraints (no barriers)")
    print(f"Stage 2 (Safety Filter): JointVelocityTask + constraints + barriers")

    print(f"\nPosition Barrier (CBF) - Stage 2 only:")
    print(f"  EE start position: {initial_ee_pos}")
    print(f"  Box min: {p_min}")
    print(f"  Box max: {p_max}")
    print(f"  Gain: {barrier_gain}, dt: {dt}, safety_margin: {safety_margin}")

    # Visualize the barrier box in Viser
    box_center = initial_ee_pos
    viz.viewer.scene.add_box(
        "/barrier_box",
        dimensions=(barrier_size, barrier_size, barrier_size),
        position=box_center,
        color=(255, 100, 100),
        opacity=0.15,
    )
    viz.viewer.scene.add_box(
        "/barrier_box_wireframe",
        dimensions=(barrier_size, barrier_size, barrier_size),
        position=box_center,
        color=(255, 50, 50),
        opacity=0.5,
        side="back",
    )

    # Get starting joint configuration
    q_canonical_raw = np.array(model_data.starting_joint_config)
    if len(q_canonical_raw) != len(q_full):
        print(
            f"\nWarning: starting_joint_config size ({len(q_canonical_raw)}) doesn't match "
            f"configuration space dimension ({len(q_full)}), using current scene positions instead"
        )
        with scene_lock:
            q_canonical = scene.getCurrentJointPositions()
    else:
        q_canonical = q_canonical_raw
    print(f"\nUsing starting pose for '{model}'")

    # Create a ConfigurationTask to regularize toward the starting pose (Stage 1)
    joint_weights = np.full(num_variables, 0.1)
    joint_weights[0] = 0.2
    if model == "dual":
        joints_per_arm = num_variables // 2
        joint_weights[joints_per_arm] = 0.2
    config_options = ConfigurationTaskOptions(task_gain=0.1, lm_damping=0.0)
    config_task = ConfigurationTask(q_canonical, joint_weights, config_options)

    # Task parameters for Stage 1 (IK)
    # No error saturation needed since we're filtering with Stage 2
    task_options = FrameTaskOptions(
        position_cost=2.0,
        orientation_cost=1.0,
        task_gain=task_gain,
        lm_damping=lm_damping,
    )

    # Safety filter task options for Stage 2
    # gain=1.0 means minimize ||dq - dq_nominal||^2 exactly
    filter_options = JointVelocityTaskOptions(task_gain=1.0, lm_damping=0.0)
    # Equal weights for all joints in the filter
    filter_weights = np.ones(num_variables)

    # Goal configuration
    goals = []
    transform_controls = []

    for name in model_data.ee_names:
        goal = CartesianConfiguration()
        goal.base_frame = model_data.base_link
        goal.tip_frame = name
        # Initialize tform to current EE pose (important!)
        goal.tform = scene.forwardKinematics(q_full, name)
        goals.append(goal)

        controls = viz.viewer.scene.add_transform_controls(
            "/ik_marker/" + name,
            depth_test=False,
            scale=0.2,
            disable_sliders=True,
            visible=True,
        )
        transform_controls.append(controls)

    def update_goals(_):
        for goal, controls in zip(goals, transform_controls):
            goal.tform = pin.SE3(
                pin.Quaternion(controls.wxyz[[1, 2, 3, 0]]), controls.position
            ).homogeneous

    for controls in transform_controls:
        controls.on_update(update_goals)

    # Control loop
    running = True

    def control_loop():
        while running:
            loop_start = time.time()

            # Build IK tasks for Stage 1
            ik_tasks = [config_task]
            for goal in goals:
                frame_task = FrameTask(
                    goal.tip_frame, goal, num_variables, task_options
                )
                ik_tasks.append(frame_task)

            with scene_lock:
                q_current = scene.getCurrentJointPositions()

                # ============================================================
                # Stage 1: Solve IK (FrameTask + constraints, NO barriers)
                # This produces a nominal delta_q that may violate barriers
                # ============================================================
                delta_q_nominal = np.zeros(num_variables)
                try:
                    oink.solve_ik(
                        ik_tasks,
                        constraints,
                        [],  # No barriers in Stage 1
                        scene,
                        delta_q_nominal,
                    )
                except RuntimeError as e:
                    print(f"Warning: IK solver (Stage 1) failed: {e}")
                    delta_q_nominal = np.zeros(num_variables)

                # ============================================================
                # Stage 2: Safety Filter (JointVelocityTask + constraints + barriers)
                # Minimizes ||dq - delta_q_nominal||^2 subject to barrier constraints
                # ============================================================
                filter_task = JointVelocityTask(
                    delta_q_nominal, filter_weights, filter_options
                )
                filter_tasks = [filter_task]

                delta_q_safe = np.zeros(num_variables)
                try:
                    oink.solve_ik(
                        filter_tasks,
                        constraints,
                        barriers,  # Barriers only in Stage 2
                        scene,
                        delta_q_safe,
                    )
                except RuntimeError as e:
                    print(f"Warning: Safety filter (Stage 2) failed: {e}")
                    delta_q_safe = np.zeros(num_variables)

                # Apply the safety-filtered command
                q_current = scene.integrate(q_current, delta_q_safe)
                scene.setJointPositions(q_current)

                # Update forward kinematics for next iteration
                for goal in goals:
                    scene.forwardKinematics(q_current, goal.tip_frame)

            viz.display(q_current)

            # Maintain control loop rate
            elapsed = time.time() - loop_start
            time.sleep(max(0, dt - elapsed))

    # Start control loop in separate thread
    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()

    # Create a marker reset button.
    reset_button = viz.viewer.gui.add_button("Reset Marker")

    @reset_button.on_click
    def reset_position(_):
        with scene_lock:
            q_current = scene.getCurrentJointPositions()
            for goal, controls in zip(goals, transform_controls):
                fk_tform = scene.forwardKinematics(q_current, goal.tip_frame)
                controls.position = fk_tform[:3, 3]
                controls.wxyz = pin.Quaternion(fk_tform[:3, :3]).coeffs()[[3, 0, 1, 2]]
                # Explicitly update goal.tform (setting controls doesn't trigger callback)
                goal.tform = fk_tform

    random_button = viz.viewer.gui.add_button("Randomize Pose")

    @random_button.on_click
    def randomize_position(_):
        with scene_lock:
            q_rand = scene.randomCollisionFreePositions()
            scene.setJointPositions(q_rand)
        reset_position(_)
        viz.display(q_rand)

    # Display the arm and marker at the starting position
    q_full = q_canonical.copy()
    with scene_lock:
        scene.setJointPositions(q_full)
    viz.display(q_full)
    reset_position(None)

    print("\n=== Running ===")
    print("Drag the marker outside the barrier box to see the safety filter in action.")
    print("The IK (Stage 1) will compute a command that violates barriers,")
    print("but the Safety Filter (Stage 2) will modify it to stay safe.")

    # Sleep forever, control loop runs in background thread
    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        running = False
        control_thread.join(timeout=1.0)


if __name__ == "__main__":
    tyro.cli(main)
