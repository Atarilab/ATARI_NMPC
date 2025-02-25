import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from ..config_abstract import MPCOptConfig
from contact_tamp.traj_opt_acados.interface.acados_helper import HPIPM_MODE

@dataclass
class MPCQuadrupedCyclic(MPCOptConfig):
    ### MPC Config
    # Time horizon (s)
    time_horizon : float = 1.
    # Number of optimization nodes
    n_nodes : int = 50
    # Replanning frequency
    replanning_freq : int = 20
    # gain on joint position for torque PD
    Kp : float = 40
    # gain on joint velocities for torque PD
    Kd : float = 5
    ### Solver Config
    # Recompile solver
    recompile: bool = True
    # Solver maximum SQP iterations
    max_iter : int = 1
    # Maximum qp iteration for one SQP step 
    max_qp_iter: int = 6

@dataclass
class MPCCyclicBiped(MPCOptConfig):
    ### MPC Config
    # Time horizon (s)
    time_horizon : float = 1.
    # Number of optimization nodes
    n_nodes : int = 80
    # Replanning frequency
    replanning_freq : int = 20
    # gain on joint position for torque PD
    Kp : float = 80
    # gain on joint velocities for torque PD
    Kd : float = 4.2
    ### Solver Config
    # Recompile solver
    recompile: bool = True
    # Solver maximum SQP iterations
    max_iter : int = 1
    # Maximum qp iteration for one SQP step 
    max_qp_iter: int = 6