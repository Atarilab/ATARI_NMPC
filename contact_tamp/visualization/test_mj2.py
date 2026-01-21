import mujoco
import mujoco.viewer
import numpy as np
import subprocess
import time
from utils import scnGeomManager

from robot_descriptions.loaders.mujoco import load_robot_description

def press_keys(keys):
    key_sequence = '+'.join(keys)
    subprocess.run(['xdotool', 'key', key_sequence])

model = load_robot_description("go2_mj_description")
data = mujoco.MjData(model)

sim_step = 0
sim_dt = 0.1
model.opt.timestep = sim_dt

trajectory_length = 100
base_trajectory = np.linspace(start=[0, 0, 0.5], stop=[1, 1, 0.5], num=trajectory_length)  # Base (x, y, z) positions
joint_trajectory = np.linspace(start=[0.0]*(model.nq-7), stop=[1.0]*(model.nq-7), num=trajectory_length)  # Joint positions

box_trajectory = np.linspace(start=[0, 0, 0.3], stop=[1, 1, 0.3], num=trajectory_length)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(0.02)
    press_keys(['Tab'])
    press_keys(['shift', 'Tab'])

    # Disable some rendering features for simplicity
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    viewer.sync()
    viewer.user_scn.ngeom = 0

    scn_geom_manager = scnGeomManager(viewer)

    p_cone1 = np.array([0, 1, 0.4])
    R_cone1 = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]])
    p_cone2 = np.array([0.0, 1, 0.4])
    R_cone2 = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    p_cone3 = np.array([0.0, 1, 0.4])
    R_cone3 = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [-1, 0, 0]])
    p_cone4 = np.array([0.0, 1, 0.4])
    R_cone4 = np.array([[0, 0, -1],
                      [0, 1, 0],
                      [1, 0, 0]])
    
    while viewer.is_running():
        if sim_step >= trajectory_length:
            sim_step = 0
            
        scn_geom_manager.clear()

        data.qpos[0:3] = base_trajectory[sim_step]
        data.qpos[7:] = joint_trajectory[sim_step]

        scn_geom_manager.add_box(box_trajectory[sim_step], [0.1,0.1,0.1], np.eye(3))
        
        scn_geom_manager.add_friction_cone(p_cone1, R_cone1, 0.5)
        scn_geom_manager.add_friction_cone(p_cone2, R_cone2, 0.5)
        scn_geom_manager.add_friction_cone(p_cone2, R_cone3, 0.5)
        scn_geom_manager.add_friction_cone(p_cone2, R_cone4, 0.5)

        scn_geom_manager.add_force_vector(p_cone1, np.array([20, 0, 0]))

        sim_step += 1
        
        mujoco.mj_kinematics(model, data)
        scn_geom_manager.update()
        viewer.sync()
        data.time += sim_dt
        
        time.sleep(0.01) # add to slow down the visualization
