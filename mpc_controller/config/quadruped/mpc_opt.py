import numpy as np
from dataclasses import dataclass
from typing import List
from ..config_abstract import MPCOptConfig

@dataclass
class MPCQuadrupedCyclic(MPCOptConfig):
    time_horizon : float = 1.
    n_nodes : int = 60
    opt_dt_scale : np.ndarray = np.array([0.5, 2.5])
    replanning_freq : int = 2
    real_time : bool = False
    opt_switch_time : bool = False
    opt_cnt_pos : bool = False
    opt_peak : bool = True
    reg_eps: float = 1.0e-6
    reg_eps_e: float = 1.0e-5