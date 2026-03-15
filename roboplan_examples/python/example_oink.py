#!/usr/bin/env python3

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
    Oink,
    PositionBarrier,
    PositionLimit,
    VelocityLimit,
)
from roboplan.viser_visualizer import ViserVisualizer


class SE3ReferenceFilter:
    """First-order filter for SE3 poses to prevent sudden jumps in reference commands.

    Smoothly interpolates between the current filtered pose and the target pose
    using exponential filtering for position and SLERP for orientation.

    This helps maintain CBF constraint validity by preventing large instantaneous
    changes in the task reference that would create large errors.
    """

    def __init__(self, tau: float = 0.1):
        """Initialize the filter.

        Args:
            tau: Time constant in seconds. Larger values = slower, smoother tracking.
                 tau=0.1 means ~63% of the step is taken per 0.1 seconds.
        """
        self.tau = tau
        self.filtered_position: np.ndarray | None = None
        self.filtered_quaternion: pin.Quaternion | None = None

    def reset(self, pose: np.ndarray) -> None:
        """Reset the filter state to a specific pose.

        Args:
            pose: 4x4 homogeneous transformation matrix.
        """
        self.filtered_position = pose[:3, 3].copy()
        self.filtered_quaternion = pin.Quaternion(pose[:3, :3])

    def update(self, target_pose: np.ndarray, dt: float) -> np.ndarray:
        """Filter the target pose and return the smoothed result.

        Args:
            target_pose: 4x4 homogeneous transformation matrix (target).
            dt: Time step in seconds.

        Returns:
            4x4 homogeneous transformation matrix (filtered).
        """
        target_position = target_pose[:3, 3]
        target_quaternion = pin.Quaternion(target_pose[:3, :3])

        # Initialize on first call
        if self.filtered_position is None:
            self.filtered_position = target_position.copy()
            self.filtered_quaternion = target_quaternion
            return target_pose.copy()

        # Exponential filter coefficient: alpha = 1 - exp(-dt/tau)
        # alpha -> 0 as tau -> inf (slower), alpha -> 1 as tau -> 0 (faster)
        alpha = 1.0 - np.exp(-dt / self.tau)

        # Filter position (linear interpolation)
        self.filtered_position = self.filtered_position + alpha * (
            target_position - self.filtered_position
        )

        # Filter orientation (SLERP)
        self.filtered_quaternion = self.filtered_quaternion.slerp(
            alpha, target_quaternion
        )

        # Construct filtered pose
        filtered_pose = np.eye(4)
        filtered_pose[:3, :3] = self.filtered_quaternion.toRotationMatrix()
        filtered_pose[:3, 3] = self.filtered_position

        return filtered_pose


def main(
    model: str = "ur5",
    task_gain: float = 1.0,
    lm_damping: float = 2.0,
    control_freq: float = 100.0,
    max_joint_velocity: float = 1.0,
    barrier_gain: float = 100.0,
    barrier_size: float = 1.0,
    safety_margin: float = 0.1,
    max_position_error: float = 0.15,
    max_rotation_error: float = 0.5,
    reference_filter_tau: float = 0.1,
    host: str = "localhost",
    port: str = "8000",
):
    """
    Run the optimal IK example with position barrier constraints.

    The barrier uses Control Barrier Functions (CBFs) to keep the end-effector
    within a bounding box. Conservative parameters (high gain, safety margin)
    ensure the barrier is never violated.

    Parameters:
        model: The name of the model to use (ur5, franka, or dual).
        task_gain: Task gain (alpha) for the IK solver (0-1).
        lm_damping: Levenberg-Marquardt damping for regularization.
        control_freq: Control loop frequency in Hz.
        max_joint_velocity: Maximum joint velocity in rad/s for all joints.
        barrier_gain: Barrier gain for CBF constraint (higher = more conservative).
        barrier_size: Size of the cubic barrier box around the EE start position (meters).
        safety_margin: Distance from boundary where barrier activates (meters).
        max_position_error: Maximum position error magnitude in meters. Prevents large
            jumps that can invalidate CBF linearization.
        max_rotation_error: Maximum rotation error magnitude in radians. Prevents large
            jumps that can invalidate CBF linearization.
        reference_filter_tau: Time constant for reference filtering in seconds. Smooths
            target pose changes to prevent sudden jumps. Set to 0 to disable filtering.
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
    print(f"\n=== Model: {model} ===")
    joint_names = scene.getJointNames()
    actuated_joint_names = scene.getActuatedJointNames()
    print(f"Total joints: {len(joint_names)}")
    print(f"Actuated joints: {len(actuated_joint_names)}")
    print(f"\nAll joint names:")
    for i, name in enumerate(joint_names):
        print(f"  {i}: {name}")
    print(f"\nActuated joint names:")
    for i, name in enumerate(actuated_joint_names):
        print(f"  {i}: {name}")
    print()

    q_full = scene.getCurrentJointPositions()

    # Create a redundant Pinocchio model just for visualization.
    # When Pinocchio 4.x releases nanobind bindings, we should be able to directly grab the model from the scene instead.
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
    # For models with quaternion joints, nv < nq
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

    # Create position limit constraint
    position_limit = PositionLimit(num_variables, gain=1.0)

    # Create velocity limit constraint
    # v_max is in rad/s, the constraint limits delta_q to [-dt*v_max, dt*v_max]
    v_max = np.full(num_variables, max_joint_velocity)
    velocity_limit = VelocityLimit(num_variables, dt, v_max)

    constraints = [position_limit, velocity_limit]

    # Get initial EE position for barrier creation
    initial_ee_pose = scene.forwardKinematics(q_full, model_data.ee_names[0])
    initial_ee_pos = initial_ee_pose[:3, 3]

    # Create position barrier with conservative parameters
    # - High gain ensures strong resistance to boundary approach
    # - Safety margin shifts the effective boundary inward to account for linearization errors
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

    print(f"\nPosition Barrier (CBF):")
    print(f"  EE start position: {initial_ee_pos}")
    print(f"  Box min: {p_min}")
    print(f"  Box max: {p_max}")
    print(f"  Gain: {barrier_gain}, dt: {dt}, safety_margin: {safety_margin}")
    print(f"  Effective boundary: {safety_margin}m inside the box")

    print(f"\nError Saturation (for CBF stability):")
    print(f"  max_position_error: {max_position_error}m")
    print(
        f"  max_rotation_error: {max_rotation_error} rad ({np.degrees(max_rotation_error):.1f} deg)"
    )

    print(f"\nReference Filtering:")
    if reference_filter_tau > 0:
        print(f"  tau: {reference_filter_tau}s (smooths target pose changes)")
    else:
        print(f"  Disabled (tau=0)")

    # Visualize the barrier box in Viser
    box_center = initial_ee_pos
    viz.viewer.scene.add_box(
        "/barrier_box",
        dimensions=(barrier_size, barrier_size, barrier_size),
        position=box_center,
        color=(255, 100, 100),
        opacity=0.15,
    )
    # Add wireframe edges for better visibility
    viz.viewer.scene.add_box(
        "/barrier_box_wireframe",
        dimensions=(barrier_size, barrier_size, barrier_size),
        position=box_center,
        color=(255, 50, 50),
        opacity=0.5,
        side="back",  # Only render back faces for wireframe effect
    )

    # Validate starting joint configuration size (should match nq)
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
    print(
        f"\nUsing starting pose for '{model}' (configuration space size: {len(q_canonical)})"
    )
    print(f"  {q_canonical}")

    # Create a ConfigurationTask to regularize toward the starting pose
    # Joint weights: 0.2 for base joints, 0.1 for all others
    joint_weights = np.full(num_variables, 0.1)
    joint_weights[0] = 0.2
    # For dual-arm, also weight the right arm's base joint symmetrically
    if model == "dual":
        joints_per_arm = num_variables // 2  # 9 joints per arm (7 arm + 2 gripper)
        joint_weights[joints_per_arm] = 0.2
    config_options = ConfigurationTaskOptions(task_gain=0.1, lm_damping=0.0)
    config_task = ConfigurationTask(q_canonical, joint_weights, config_options)

    # Task parameters (define before using in control loop)
    # max_position_error and max_rotation_error prevent large error jumps that can
    # invalidate the linearized CBF constraint, improving barrier stability.
    task_options = FrameTaskOptions(
        position_cost=2.0,
        orientation_cost=1.0,
        task_gain=task_gain,
        lm_damping=lm_damping,
        max_position_error=max_position_error,
        max_rotation_error=max_rotation_error,
    )

    # Goal configuration
    goals = []
    transform_controls = []
    reference_filters = []  # One filter per EE for smooth reference tracking
    raw_target_poses = []  # Unfiltered target poses from the controls

    # First, create all goals and controls
    for name in model_data.ee_names:
        goal = CartesianConfiguration()
        goal.base_frame = model_data.base_link
        goal.tip_frame = name
        goals.append(goal)

        # Create an interactive marker
        controls = viz.viewer.scene.add_transform_controls(
            "/ik_marker/" + name,
            depth_test=False,
            scale=0.2,
            disable_sliders=True,
            visible=True,
        )
        transform_controls.append(controls)

        # Create a reference filter for this EE (if tau > 0)
        if reference_filter_tau > 0:
            reference_filters.append(SE3ReferenceFilter(tau=reference_filter_tau))
        else:
            reference_filters.append(None)  # No filtering

        # Initialize raw target pose storage
        raw_target_poses.append(np.eye(4))

    # Lock for raw target poses (updated by UI callback, read by control loop)
    raw_targets_lock = threading.Lock()

    # Now set up the callback after all controls are created
    def update_goals(_):
        # Store raw target poses from the marker positions (thread-safe)
        with raw_targets_lock:
            for i, controls in enumerate(transform_controls):
                raw_target_poses[i] = pin.SE3(
                    pin.Quaternion(controls.wxyz[[1, 2, 3, 0]]), controls.position
                ).homogeneous

    # Attach the callback to all controls
    for controls in transform_controls:
        controls.on_update(update_goals)

    # Control loop
    running = True

    def control_loop():
        while running:
            loop_start = time.time()

            # Get raw target poses (thread-safe)
            with raw_targets_lock:
                current_raw_targets = [pose.copy() for pose in raw_target_poses]

            # Build tasks with filtered targets
            current_tasks = [config_task]
            for i, (goal, raw_target, ref_filter) in enumerate(
                zip(goals, current_raw_targets, reference_filters)
            ):
                # Apply reference filtering if enabled
                if ref_filter is not None:
                    filtered_target = ref_filter.update(raw_target, dt)
                else:
                    filtered_target = raw_target

                # Update goal with filtered pose
                goal.tform = filtered_target

                # Create a FrameTask for this goal
                frame_task = FrameTask(
                    goal.tip_frame, goal, num_variables, task_options
                )
                current_tasks.append(frame_task)

            # Solve IK
            with scene_lock:
                # Get current joint configuration
                q_current = scene.getCurrentJointPositions()

                # Solve IK for one step with constraints and barriers
                delta_q = np.zeros(num_variables)
                try:
                    oink.solve_ik(current_tasks, constraints, barriers, scene, delta_q)
                except RuntimeError as e:
                    print(f"Warning: IK solver failed: {e}")

                # Integrate: delta_q is a displacement (already limited by VelocityLimit)
                q_current = scene.integrate(q_current, delta_q)

                # Update scene state
                scene.setJointPositions(q_current)

                # Update forward kinematics after applying velocities
                # This ensures FK is current for the next iteration's solve_ik
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
            for i, (goal, controls, ref_filter) in enumerate(
                zip(goals, transform_controls, reference_filters)
            ):
                fk_tform = scene.forwardKinematics(q_current, goal.tip_frame)
                controls.position = fk_tform[:3, 3]
                controls.wxyz = pin.Quaternion(fk_tform[:3, :3]).coeffs()[[3, 0, 1, 2]]

                # Reset the reference filter to the current pose
                if ref_filter is not None:
                    ref_filter.reset(fk_tform)

                # Update raw target pose
                with raw_targets_lock:
                    raw_target_poses[i] = fk_tform.copy()

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

    # Sleep forever, control loop runs in background thread
    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        running = False
        control_thread.join(timeout=1.0)


if __name__ == "__main__":
    tyro.cli(main)
