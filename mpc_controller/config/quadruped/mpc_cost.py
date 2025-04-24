import numpy as np
from dataclasses import dataclass
from typing import Any, List
from ..config_abstract import MPCCostConfig
import numpy as np
from dataclasses import dataclass, field
from ..config_abstract import MPCCostConfig

HIP_SHOULDER_ELBOW_SCALE = [15., 5., 1.]
# PENALIZE JOINT MOTION
W_JOINT = 1.
N_FEET = 4

@dataclass
class Go2TrotCost(MPCCostConfig):
    @staticmethod
    def __init_np(l : List, scale : float=1.):
        """ Init numpy array field."""
        return field(default_factory=lambda: np.array(l) * scale)

    # Robot name
    robot_name: str = "Go2"
    gait_name: str = "trot"

    # Updated base running cost weights
    W_base: np.ndarray = __init_np([
        1e3, 3e3, 1e2,      # Base position weights
        5e2, 5e2, 5e2,      # Base orientation (ypr) weights
        5e2, 1e1, 1e0,      # Base linear velocity weights
        1e0, 2e1, 1e1,      # Base angular velocity weights
    ], 1.)

    # Updated base terminal cost weights
    W_e_base: np.ndarray = __init_np([
        1e1, 1e1, 1e3,      # Base position weights
        1e1, 1e2, 1e2,      # Base orientation (ypr) weights
        5e2, 5e2, 1e3,      # Base linear velocity weights
        1e1, 1e2, 1e2,      # Base angular velocity weights
    ], 1)

    # Joint running cost to nominal position and vel (hip, shoulder, elbow)
    W_joint: np.ndarray = __init_np(HIP_SHOULDER_ELBOW_SCALE * N_FEET + [0.03] * len(HIP_SHOULDER_ELBOW_SCALE) * N_FEET, 5.)

    # Joint terminal cost to nominal position and vel (hip, shoulder, elbow)
    W_e_joint: np.ndarray = __init_np(HIP_SHOULDER_ELBOW_SCALE * N_FEET  + [0.1] * len(HIP_SHOULDER_ELBOW_SCALE) * N_FEET, 1.)

    # Acceleration cost weights for joints (hip, shoulder, elbow)
    W_acc: np.ndarray = __init_np(HIP_SHOULDER_ELBOW_SCALE * N_FEET, 5.0e-4)

    # swing cost weightsc
    W_swing: np.ndarray = __init_np([2e4] * N_FEET)

    # End effector orientation wrt normal at contact
    W_eeff_ori: np.ndarray = __init_np([1.] * N_FEET)

    # force regularization weights for each foot
    W_cnt_f_reg: np.ndarray = __init_np([[0.01, 0.01, 0.05]] * N_FEET)

    # Feet position constraint stability
    W_foot_pos_constr_stab: np.ndarray = __init_np([5e1] * N_FEET)

    # Foot displacement penalization
    W_foot_displacement: np.ndarray = __init_np([1e3])

    # Contact restriction radius
    cnt_radius: float = 0.015 # m

    # Time opt cost
    time_opt: np.ndarray = __init_np([1.0e4])

    reg_eps: float = 1.0e-6
    reg_eps_e: float = 1.0e-5

W = [
        0e0, 0e0, 5e3,      # Base position weights
        0e0, 3e3, 3e3,      # Base orientation (ypr) weights
        0e0, 0e0, 1e1,      # Base linear velocity weights
        1e0, 1e2, 2e2,      # Base angular velocity weights
    ]

@dataclass
class Go2SlowTrotCost(MPCCostConfig):
    @staticmethod
    def __init_np(l : List, scale : float=1.):
        """ Init numpy array field."""
        return field(default_factory=lambda: np.array(l) * scale)

    # Robot name
    robot_name: str = "Go2"
    gait_name: str = "slow_trot"

    # Updated base running cost weights
    W_base: np.ndarray = __init_np(W, 7.)

    # Updated base terminal cost weights
    W_e_base: np.ndarray = __init_np(W, 10.)

    # Joint running cost to nominal position and vel (hip, shoulder, elbow)
    W_joint: np.ndarray = __init_np(HIP_SHOULDER_ELBOW_SCALE * N_FEET + [0.] * len(HIP_SHOULDER_ELBOW_SCALE) * N_FEET, 0.1)

    # Joint terminal cost to nominal position and vel (hip, shoulder, elbow)
    W_e_joint: np.ndarray = __init_np(HIP_SHOULDER_ELBOW_SCALE * N_FEET + [0] * len(HIP_SHOULDER_ELBOW_SCALE) * N_FEET, 0.)

    # Acceleration cost weights for joints (hip, shoulder, elbow)
    W_acc: np.ndarray = __init_np([7., 3., 1.] * N_FEET, 1.e-2)

    # swing cost weightsc
    W_swing: np.ndarray = __init_np([5e5] * N_FEET)
    
    # End effector orientation wrt normal at contact
    W_eeff_ori: np.ndarray = __init_np([0.] * N_FEET)

    # force regularization weights for each foot
    W_cnt_f_reg: np.ndarray = __init_np([[1.2, 1.2, 0.9]] * N_FEET, 1.)

    # Feet position constraint stability
    W_foot_pos_constr_stab: np.ndarray = __init_np([5e1] * N_FEET)

    # Foot displacement penalization
    W_foot_displacement: np.ndarray = __init_np([1e6])

    # Contact restriction radius
    cnt_radius: float = 0.005 # m

    # Time opt cost
    time_opt: np.ndarray = __init_np([1.0e4])

    reg_eps: float = 1.0e-6
    reg_eps_e: float = 1.0e-5

class CostConfigFactory():
    AVAILABLE_COST = {
        (cfg.robot_name.lower(),  cfg.gait_name.lower()) : cfg()
        for cfg in [
            Go2TrotCost,
            Go2SlowTrotCost,
        ]
    }

    @staticmethod
    def get(robot_name: str, gait_name: str) -> MPCCostConfig:
        config = CostConfigFactory.AVAILABLE_COST.get((robot_name.lower(), gait_name.lower()), None)
        if config is None:
            raise ValueError(f"Cost config: {gait_name} for {robot_name} not available.")
        return config
