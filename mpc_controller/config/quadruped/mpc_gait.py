from dataclasses import dataclass, field
import numpy as np
from ..config_abstract import GaitConfig

@dataclass
class QuadrupedGaitConfig(GaitConfig):
    n_eeff: int = 4  # Number of end-effectors (4 legs)
    
    def __post_init__(self):
        super().__post_init__()
        assert len(self.stance_ratio) == self.n_eeff, f"stance_ratio must be of length {self.n_eeff}"
        assert len(self.phase_offset) == self.n_eeff, f"phase_offset must be of length {self.n_eeff}"

@dataclass
class QuadrupedTrot(QuadrupedGaitConfig):
    gait_name: str = "trot"
    nominal_period: float = 0.5
    stance_ratio: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.5]))
    phase_offset: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 0.0, 0.5]))
    nom_height: float = 0.30
    step_height: float = 0.05

@dataclass
class QuadrupedSlowTrot(QuadrupedGaitConfig):
    gait_name: str = "slow_trot"
    nominal_period: float = 1.
    stance_ratio: np.ndarray = field(default_factory=lambda: np.array([0.63, 0.63, 0.63, 0.63]))
    phase_offset: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 0.0, 0.5]))
    nom_height: float = 0.32
    step_height: float = 0.065

@dataclass
class QuadrupedJump(QuadrupedGaitConfig):
    gait_name: str = "jump"
    nominal_period : float = 50
    stance_ratio: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.4, 0.4]))
    phase_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0]))
    nom_height: float = 0.3
    step_height: float = 0.05

@dataclass
class QuadrupedCrawl(QuadrupedGaitConfig):
    gait_name: str = "crawl"
    nominal_period : float = 1.
    stance_ratio: np.ndarray = field(default_factory=lambda: np.array([0.75, 0.75, 0.75, 0.75]))
    phase_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.25, 0.5, 0.75]))
    nom_height: float = 0.3
    step_height: float = 0.05

@dataclass
class QuadrupedPace(QuadrupedGaitConfig):
    gait_name: str = "pace"
    nominal_period : float = 0.5
    stance_ratio: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.6, 0.6, 0.6]))
    phase_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.5, 0.5, 0.0]))
    nom_height: float = 0.05
    step_height: float = 0.32

@dataclass
class QuadrupedBound(QuadrupedGaitConfig):
    gait_name: str = "bound"
    nominal_period : float = 0.5
    stance_ratio: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.6, 0.6, 0.6]))
    phase_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.5, 0.5, 0.0]))
    nom_height: float = 0.05
    step_height: float = 0.32

class GaitConfigFactory:
    AVAILABLE_GAITS = {
        cfg.gait_name.lower(): cfg()
        for cfg in [
            QuadrupedTrot,
            QuadrupedSlowTrot,
            QuadrupedJump,
            QuadrupedCrawl,
            QuadrupedPace,
            QuadrupedBound,
        ]
    }

    @staticmethod
    def get(gait_name: str) -> GaitConfig:
        config = GaitConfigFactory.AVAILABLE_GAITS.get(gait_name.lower(), None)
        if config is None:
            raise ValueError(f"{gait_name} not available.")
        return config