from typing import Tuple

from ..config_abstract import GaitConfig, MPCOptConfig, MPCCostConfig
from .mpc_gait import GaitConfigFactory
from .mpc_opt import MPCQuadrupedCyclic, MPCCyclicBiped
from .mpc_cost import CostConfigFactory

def get_quadruped_config(
        gait_name: str,
        robot_name : str,
        ) -> Tuple[GaitConfig, MPCOptConfig, MPCCostConfig]:

        gait_config = GaitConfigFactory.get(gait_name)
        cost_config = CostConfigFactory.get(robot_name, gait_name)
        opt_config = MPCQuadrupedCyclic()
        
        return gait_config, opt_config, cost_config

def get_biped_config(
        gait_name: str,
        robot_name : str,
        ) -> Tuple[GaitConfig, MPCOptConfig, MPCCostConfig]:

        gait_config = GaitConfigFactory.get(gait_name)
        cost_config = CostConfigFactory.get(robot_name,gait_name)
        opt_config = MPCCyclicBiped()
        
        return gait_config, opt_config, cost_config