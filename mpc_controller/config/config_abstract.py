import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field
from typeguard import typechecked
from contact_tamp.traj_opt_acados.interface.acados_helper import HPIPM_MODE

@dataclass
@typechecked
class GaitConfig():
    # Name of the gait
    gait_name : str
    # Nominal gait period (s)
    nominal_period : float
    # Gait stance ratio in [0,1]
    stance_ratio : np.ndarray
    # Gait offset between legs in [0,1]
    phase_offset : np.ndarray
    # Gait nominal height (m)
    nom_height : float
    # Gait step height (m)
    step_height : float

    def __post_init__(self):
        assert  (all(0 <= r <= 1 for r in self.stance_ratio)), "stance_ratio should be in [0,1]"
        assert  (all(0 <= r <= 1 for r in self.phase_offset)), "phase_offset should be in [0,1]"

@dataclass
@typechecked
class MPCOptConfig():
    # Time horizon (s)
    time_horizon : float
    # Number of optimization nodes
    n_nodes : int
    # Replanning frequency
    replanning_freq : int
    # gain on joint position for torque PD
    Kp : float
    # gain on joint velocities for torque PD
    Kd : float
    # Recompile solver
    recompile: bool
    # Solver maximum SQP iterations
    max_iter : int
    # Maximum qp iteration for one SQP step 
    max_qp_iter: int
    # Real time iterations
    real_time_it : bool = False
    # Enable dt nodes time optimization
    enable_time_opt : bool = False
    # Time bounds between two nodes (min_ratio, max_ratio)
    # dt_min = min_ratio * dt_nom = min_ratio * (time_horizon / n_nodes)
    opt_dt_scale : Tuple[float, float] = (0.5, 1.75)
    # Enable impact dynamics
    enable_impact_dyn : bool = False
    # Use peak constrained
    opt_peak : bool = True
    # Warm start states and inputs with last solution
    warm_start_sol : bool = True
    # Warm start solver IP outer loop
    warm_start_nlp : bool = True
    # Warm start solver IP inner loop
    warm_start_qp : bool = True
    # HPIPM mode
    hpipm_mode : HPIPM_MODE = HPIPM_MODE.speed
    # use_cython in the solver solver
    use_cython: bool = False

    # Outer loop SQP tolerance
    nlp_tol: float = 1.e-1
    # Inner loop interior point method tolerance
    qp_tol: float = 1.e-2

    def __post_init__(self):
        assert len(self.opt_dt_scale) == 2, "opt_dt_scale must be of shape 2"

    def get_dt_bounds(self) -> Tuple[float, float]:
        """
        Return optimization bounds for the time between two optimization nodes.
        """
        dt_nodes = self.get_dt_nodes()
        return (
            round(dt_nodes * self.opt_dt_scale[0], 4),
            round(dt_nodes * self.opt_dt_scale[1], 4),
        )
    
    def get_dt_nodes(self) -> float:
        """
        Return nominal time between two optimization nodes.
        """
        return round(self.time_horizon / self.n_nodes, 4)

@dataclass
@typechecked
class MPCCostConfig:
    robot_name: str
    gait_name: str
    # Weights for the terminal cost of base position, orientation, and velocity
    # [x, w, z, ox, oy, oz, vx, vy, vz, wx, wy, wz]
    W_e_base: np.ndarray
    # Weights for the running cost of base position, orientation, and velocity
    # [x, w, z, ox, oy, oz, vx, vy, vz, wx, wy, wz]
    W_base: np.ndarray
    # Weights for the running cost of joint to nominal joint pos and vel
    # [hip_FL, shoulder_FL, elbow_FL, ...FR, RL, RR] [pos, vel]
    W_joint: np.ndarray
    # Weights for the terminal cost of joint to nominal joint pos and vel
    # [hip_FL, shoulder_FL, elbow_FL, ...FR, RL, RR] [pos, vel]
    W_e_joint: np.ndarray
    # Weights for joint acceleration cost
    # [hip_FL, shoulder_FL, elbow_FL, ...FR, RL, RR]
    W_acc: np.ndarray
    # Weights for swing cost (eeff motion)
    # length: number of eeff
    W_swing: np.ndarray
    # Weights for force regularization (per foot) [[x, y, z], ...]
    # length: number of eeff
    W_cnt_f_reg: np.ndarray
    # Weight constraint contact horizontal velocity
    W_foot_pos_constr_stab: np.ndarray
    # Foot displacement from target penalization
    W_foot_displacement: np.ndarray
    # Default contact radius for contact restriction
    cnt_radius: float
    # Time opt cost
    time_opt: np.ndarray
    # Reguralization running cost
    reg_eps: float
    # Reguralization terminal cost
    reg_eps_e: float
    
    def __post_init__(self):
        assert len(self.W_e_base) == 12, "W_e_base must be of shape 12"
        assert len(self.W_base) == 12, "W_base must be of shape 12"
        assert (len(self.W_swing) == len(self.W_cnt_f_reg) and
                len(self.W_swing) == len(self.W_foot_pos_constr_stab)),\
                "W_swing and W_foot should have the same length."
        # Wrench Change        
        for i, foot_weight in enumerate(self.W_cnt_f_reg):
            assert len(foot_weight) == 6, f"W_foot[{i}] must be of shape 6"
