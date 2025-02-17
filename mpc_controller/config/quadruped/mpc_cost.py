import numpy as np
from dataclasses import dataclass
from typing import Any, List
from ..config_abstract import MPCCostConfig
import numpy as np
from dataclasses import dataclass, field
from ..config_abstract import MPCCostConfig

HIP_SHOULDER_ELBOW_SCALE = [15., 5., 1.]
PELV_HIP_KNEE_ANKLE_SCALE = [5., 5., 5., 2.]
# PENALIZE JOINT MOTION
W_JOINT = 1.
N_FEET = 4
N_EEFF = 2 # Number of biped end-effectors(legs)

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
        1e3, 1e3, 5e4,      # Base position weights
        1e5, 1e5, 1e5,      # Base orientation (ypr) weights
        1e5, 1e5, 5e3,      # Base linear velocity weights
        1e4, 1e3, 1e3,      # Base angular velocity weights
    ])

    # Updated base terminal cost weights
    W_e_base: np.ndarray = __init_np([
        1e5, 1e5, 1e5,     # Base position weights
        1e4, 1e4, 1e4,     # Base orientation (ypr) weights
        1e3, 1e3, 1e3,     # Base linear velocity weights
        1e3, 1e3, 1e3      # Base angular velocity weights
    ])

    # Joint running cost to nominal position and vel (hip, shoulder, elbow)
    W_joint: np.ndarray = __init_np(HIP_SHOULDER_ELBOW_SCALE * N_FEET + [0.5] * len(HIP_SHOULDER_ELBOW_SCALE) * N_FEET, W_JOINT)

    # Joint terminal cost to nominal position and vel (hip, shoulder, elbow)
    W_e_joint: np.ndarray = __init_np(HIP_SHOULDER_ELBOW_SCALE * N_FEET + [0] * len(HIP_SHOULDER_ELBOW_SCALE) * N_FEET, W_JOINT)

    # Acceleration cost weights for joints (hip, shoulder, elbow)
    W_acc: np.ndarray = __init_np(HIP_SHOULDER_ELBOW_SCALE * N_FEET, 5.0e-3)

    # swing cost weightsc
    W_swing: np.ndarray = __init_np([1e6] * N_FEET)

    # force regularization weights for each foot
    W_cnt_f_reg: np.ndarray = __init_np([[0.75, 0.75, 5e-1]] * N_FEET)

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

    # force regularization weights for each foot
    W_cnt_f_reg: np.ndarray = __init_np([[1.25, 1.25, 0.9]] * N_FEET, 1.)

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



@dataclass
class G1CyclicCost(MPCCostConfig):
    @staticmethod
    def __init_np(l : List, scale : float=1.):
        """ Init numpy array field."""
        return field(default_factory=lambda: np.array(l) * scale)

    # Robot name
    robot_name: str = "g1"
    gait_name: str = "trot_biped"
    # Updated base running cost weights

    # Updated base running cost weights
    W_base: np.ndarray = __init_np([
        1e3, 1e3, 5e6,      # Base position weights
        1e6, 5e7, 5e6,      # Base orientation (ypr) weights
        1e4, 1e4, 5e3,      # Base linear velocity weights
        1e6, 5e6, 5e6,      # Base angular velocity weights
    ])

    # Updated base terminal cost weights
    W_e_base: np.ndarray = __init_np([
        1e5, 1e5, 1e6,     # Base position weights
        1e6, 1e6, 1e6,     # Base orientation (ypr) weights
        0e0, 0e0, 1e3,     # Base linear velocity weights
        0e0, 0e0, 0e0      # Base angular velocity weights
    ])
    # Joint running cost to nominal position and vel (hip, shoulder, elbow)
    W_joint: np.ndarray = __init_np(PELV_HIP_KNEE_ANKLE_SCALE * N_EEFF + [0.1] * 4 * N_EEFF, W_JOINT)

    # Joint terminal cost to nominal position and vel (hip, shoulder, elbow)
    W_e_joint: np.ndarray = __init_np(PELV_HIP_KNEE_ANKLE_SCALE * N_EEFF + [0] * 4 * N_EEFF, W_JOINT)

    # Acceleration cost weights for joints (hip, shoulder, elbow)
    W_acc: np.ndarray = __init_np(PELV_HIP_KNEE_ANKLE_SCALE * N_EEFF, 5.0e-3)

    # swing cost weightsc
    W_swing: np.ndarray = __init_np([1e6] * N_EEFF)

    # force regularization weights for each foot

    W_cnt_f_reg: np.ndarray = __init_np([[1e0, 1e0, 5e-1]] * N_EEFF)
    # Feet position constraint stability
    W_foot_pos_constr_stab: np.ndarray = __init_np([5e1] * N_EEFF)

    # Foot displacement penalization
    W_foot_displacement: np.ndarray = __init_np([1e3])

    # Contact restriction radius
    cnt_radius: float = 0.015 # m

    # Time opt cost
    time_opt: np.ndarray = __init_np([1.0e4])

    reg_eps: float = 1.0e-6
    reg_eps_e: float = 1.0e-5



class CostConfigFactory():
    AVAILABLE_COST = {
        (cfg.robot_name.lower(),  cfg.gait_name.lower()) : cfg()
        for cfg in [
            # Go2TrotCost,
            # Go2SlowTrotCost,
            G1CyclicCost
        ]
    }

    @staticmethod
    def get(robot_name: str, gait_name: str) -> MPCCostConfig:
        config = CostConfigFactory.AVAILABLE_COST.get((robot_name.lower(), gait_name.lower()), None)
        if config is None:
            raise ValueError(f"Cost config: {gait_name} for {robot_name} not available.")
        return config
