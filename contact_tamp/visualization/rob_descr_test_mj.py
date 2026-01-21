import mujoco
import mujoco.viewer
import numpy as np
import time

from robot_descriptions.loaders.mujoco import load_robot_description

def mj_actuator_limits(model):
    for i in range(model.nu):  # model.nu gives the number of actuators
        # Get the actuator name
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)

        # Check for effort limits
        if model.actuator_forcelimited[i]:
            effort_limit = model.actuator_ctrlrange[i]  # Control range gives force/torque limits
        else:
            effort_limit = "No limit"

        # Check if actuator controls a joint
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            # Get the joint ID the actuator is controlling
            joint_id = model.actuator_trnid[i, 0]

            # Ensure the joint_id is valid and within range
            if joint_id < model.njnt:
                pos_range = model.jnt_range[joint_id]
            else:
                pos_range = "Invalid joint ID"
        else:
            pos_range = "No limit"

        # Print actuator information
        print(f"Actuator {i+1}: {actuator_name}")
        print(f"Effort Limit: {effort_limit}  Pos Limit: {pos_range}")

if __name__ == "__main__":
    
    model = load_robot_description("go2_mj_description")
    data = mujoco.MjData(model)

    sim_step = 0
    sim_dt = 0.1
    model.opt.timestep = sim_dt

    base_trajectory = np.linspace(start=[0, 0, 0.5], stop=[1, 1, 0.5], num=100)  # Base (x, y, z) positions
    joint_trajectory = np.linspace(start=[0.0]*(model.nq-7), stop=[1.0]*(model.nq-7), num=100)  # Joint positions

    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Disable some rendering features for simplicity
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        viewer.sync()

        print(f"Number of actuators: {model.nu}")
        mj_actuator_limits(model)

        while viewer.is_running():
            data.qpos[0:3] = base_trajectory[sim_step]
            data.qpos[7:] = joint_trajectory[sim_step]
            sim_step += 1
            mujoco.mj_kinematics(model, data)

            viewer.sync()
            data.time += sim_dt
            time.sleep(sim_dt)
