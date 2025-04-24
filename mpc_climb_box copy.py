import numpy as np
from typing import Any, List
import mujoco
import os

from sdk_controller.robots import Go2

from mj_pin.utils import get_robot_description
from mj_pin.simulator import Simulator
from mpc_controller.config.config_abstract import MPCOptConfig, MPCCostConfig, GaitConfig, HPIPM_MODE
from mpc_controller.mpc_acyclic import AcyclicMPC
from search.mcts_locomotion_task import MCTSPhaseLocomotionTask
from scene.climb_box import setup_scene, SCENE_NAME

BASE_SAVE_DIR = "data"
# SIM
ROBOT_NAME = Go2.ROBOT_NAME
SIM_DT = 1 / Go2.CONTROL_FREQ
# CLOSE LOOP MPC
DT = 0.04
N_NODES_MPC = 35
# TRAJ OPT
RECOMPILE = False
N_NODES_SOLVER = 50
MAX_IT = 75
MAX_QP = 6
# SCENE PARAM
HEIGHT = 0.24
EDGE = 0.48
OFFSET = 0.5

robot_description = get_robot_description(ROBOT_NAME)
mj_feet_frames = ["FL", "FR", "RL", "RR"]
pin_feet_frames = [f + "_foot" for f in mj_feet_frames]
n_feet = len(mj_feet_frames)
sim = Simulator(robot_description.xml_scene_path)

################# Setup scene task
surfaces = setup_scene(sim, height=HEIGHT, edge=EDGE, offset=OFFSET, save_dir="", vis_normal=False)
surfaces = surfaces[:2]


sim._init_model_data()
mj_spec = sim.edit.mj_spec
# Save mj_spec to an XML file
xml_save_path = os.path.join(f"scene/{SCENE_NAME}.xml")
with open(xml_save_path, "w") as xml_file:
    xml_file.write(mj_spec.to_xml())
    
##################  Solver
# Opt
config_close_loop = MPCOptConfig(
    time_horizon=N_NODES_MPC * DT,
    n_nodes=N_NODES_MPC,
    replanning_freq=20,
    Kp=35,
    Kd=7.,
    recompile=RECOMPILE,
    max_iter=MAX_IT,
    max_qp_iter=MAX_QP,
    opt_peak=True,
    warm_start_sol=True,
    nlp_tol=1.0e-2,
    qp_tol=1.0e-3,
    hpipm_mode=HPIPM_MODE.speed,
)

# Cost
def __init_np(l : List, scale : float=1.):
    """ Init numpy array field."""
    return np.array(l) * scale

W = [
        0e0, 0e0, 1e-1,      # Base position weights
        1e-1, 3e1, 4e1,      # Base orientation (ypr) weights
        1e0, 1e0, 1e0,      # Base linear velocity weights
        1e-1, 5e0, 5e0,      # Base angular velocity weights
    ]

W_e = [
        0e0, 0e0, 1e1,      # Base position weights
        0e0, 1e1, 1e1,      # Base orientation (ypr) weights
        1e0, 1e0, 0e0,      # Base linear velocity weights
        1e-1, 1e0, 1e0,      # Base angular velocity weights
    ]

HSE_SCALE = [15., 10., 1.] *  n_feet
config_cost = MPCCostConfig(
    robot_name=ROBOT_NAME,
    gait_name="",
    W_e_base=__init_np(W_e, 10.),
    W_base=__init_np(W, 300.),
    W_joint=__init_np(HSE_SCALE + [0.05] * len(HSE_SCALE), 10.),
    W_e_joint=__init_np(HSE_SCALE + [0.01] * len(HSE_SCALE), 1.),
    W_acc=__init_np(HSE_SCALE, 1.e-3),
    W_swing=__init_np([2e4] * n_feet),
    W_eeff_ori=__init_np([1e1] * n_feet),
    W_cnt_f_reg = __init_np([[0.075, 0.075, 0.05]] * n_feet),
    W_foot_pos_constr_stab = __init_np([1e3] * n_feet),
    W_foot_displacement = __init_np([0.]),
    cnt_radius = 0.015, # m
    time_opt = __init_np([1.0e4]),
    reg_eps = 1.0e-6,
    reg_eps_e = 1.0e-5,
)

config_gait = GaitConfig(
    "acyclic",
    1.,
    np.array([0.1, 0.1, 0.1, 0.1]),
    np.array([0.1, 0.1, 0.1, 0.1]),
    0.3 + HEIGHT,
    0.075,
)

mpc_close_loop = AcyclicMPC(
    robot_description.urdf_path,
    pin_feet_frames,
    config_close_loop,
    config_cost,
    config_gait,
    joint_ref=robot_description.q0,
    sim_dt=SIM_DT,
    height_offset=0.,
    print_info=False,
    compute_timings=True,
    solve_async=True,
)
mpc_close_loop.config_opt.recompile = False