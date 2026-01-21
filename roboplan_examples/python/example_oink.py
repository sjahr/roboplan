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
from roboplan.optimal_ik import FrameTask, FrameTaskParams, Oink, PositionLimit
from roboplan.viser_visualizer import ViserVisualizer


def main(
    model: str = "ur5",
    task_gain: float = 1.0,
    lm_damping: float = 2.0,
    host: str = "localhost",
    port: str = "8000",
):
    """
    Run the optimal IK example with the provided parameters.

    Parameters:
        model: The name of the model to use (ur5, franka, or dual).
        task_gain: Task gain (alpha) for the IK solver (0-1).
        lm_damping: Levenberg-Marquardt damping for regularization.
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
    q_indices = scene.getJointGroupInfo(model_data.default_joint_group).q_indices

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

    # Set up the Oink solver with position limit constraints
    num_variables = len(q_indices)
    oink = Oink(num_variables)

    # Create position limit constraint
    position_limit = PositionLimit(num_variables, gain=1.0)
    constraints = [position_limit]

    # Goal configuration
    goal = CartesianConfiguration()
    goal.base_frame = model_data.base_link
    goal.tip_frame = model_data.ee_names[0]

    # Thread-safe task list
    tasks_lock = threading.Lock()
    tasks = []

    # Create an interactive marker.
    controls = viz.viewer.scene.add_transform_controls(
        "/ik_marker/",
        depth_test=False,
        scale=0.2,
        disable_sliders=True,
        visible=True,
    )

    @controls.on_update
    def update_goal(_):
        # Set the goal from the marker position
        goal.tform = pin.SE3(
            pin.Quaternion(controls.wxyz[[1, 2, 3, 0]]), controls.position
        ).homogeneous

        # Create a FrameTask for this goal
        params = FrameTaskParams(
            task_weight=1.0,
            position_cost=1.0,
            orientation_cost=1.0,
            task_gain=task_gain,
            lm_damping=lm_damping,
        )
        frame_task = FrameTask(goal.tip_frame, goal, params)

        # Thread-safe update of tasks
        with tasks_lock:
            tasks.clear()
            tasks.append(frame_task)

    # Control loop running at 100Hz
    running = True

    def control_loop():
        dt = 1.0 / 100.0  # 100Hz
        while running:
            loop_start = time.time()

            # Get current joint configuration
            q_current = scene.getCurrentJointPositions()[q_indices]

            # Thread-safe task access
            with tasks_lock:
                current_tasks = tasks.copy()

            # Only solve if we have tasks
            if current_tasks:
                # Solve IK for one step with constraints
                try:
                    barriers = []  # No barriers for this example
                    delta_q = oink.solveIk(current_tasks, constraints, barriers, scene)
                except RuntimeError as e:
                    print(f"Warning: IK solver failed: {e}, using zero delta_q")
                    delta_q = np.zeros(num_variables)

                # Integrate: update current config by adding delta_q
                q_current = q_current + delta_q

                # Update scene and visualization
                q_full[q_indices] = q_current
                scene.setJointPositions(q_full)
                viz.display(q_full)

                # Print goal position vs achieved position
                achieved_tform = scene.forwardKinematics(q_full, goal.tip_frame)
                goal_pos = goal.tform[:3, 3]
                achieved_pos = achieved_tform[:3, 3]
                position_error = np.linalg.norm(goal_pos - achieved_pos)

            # Maintain 100Hz loop rate
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Start control loop in separate thread
    control_thread = threading.Thread(target=control_loop, daemon=True)
    control_thread.start()

    # Create a marker reset button.
    reset_button = viz.viewer.gui.add_button("Reset Marker")

    @reset_button.on_click
    def reset_position(_):
        fk_tform = scene.forwardKinematics(
            scene.getCurrentJointPositions(), goal.tip_frame
        )
        controls.position = fk_tform[:3, 3]
        controls.wxyz = pin.Quaternion(fk_tform[:3, :3]).coeffs()[[3, 0, 1, 2]]

    random_button = viz.viewer.gui.add_button("Randomize Pose")

    @random_button.on_click
    def randomize_position(_):
        q_full = scene.getCurrentJointPositions()
        q_rand = scene.randomCollisionFreePositions()[q_indices]
        q_full[q_indices] = q_rand
        scene.setJointPositions(q_full)
        viz.display(q_full)
        reset_position(_)

    # Display the arm and marker at the starting position
    q_full[q_indices] = np.array(model_data.starting_joint_config)[q_indices]
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
