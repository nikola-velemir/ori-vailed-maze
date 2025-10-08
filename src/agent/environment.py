from typing import Tuple, Set

import pomdp_py

from src.domain.better_observation_model import BetterObservationModel
from src.models.observation_model import ObservationModel
from src.models.reward_model import RewardModel
from src.models.transition_model import TransitionModel


class MazeEnvironment(pomdp_py.Environment):
    def __init__(self,
                 width,
                 height,
                 init_state,
                 walls: Set[Tuple[int, int]],
                 traps: Set[Tuple[int, int]],
                 holes: Set[Tuple[int, int]],
                 goal_state: Tuple[int, int],
                 transition_model: TransitionModel,
                 observation_model: BetterObservationModel,
                 reward_model: RewardModel,
                 noise: float = 0.15):
        self.width = width
        self.height = height
        self.walls = walls
        self.traps = traps
        self.holes = holes
        self.noise = noise
        self.goal_state = goal_state
        self.observation_model = observation_model

        super().__init__(init_state,
                         transition_model,  reward_model)
