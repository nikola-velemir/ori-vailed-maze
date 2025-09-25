from typing import Tuple, Set

import pomdp_py

from src.domain.maze_state import MazeState
from src.models.observation_model import ObservationModel
from src.models.reward_model import RewardModel
from src.models.transition_model import TransitionModel
from src.models.policy_model import PolicyModel  # assuming you have this


class MazeProblem(pomdp_py.POMDP):
    """Complete POMDP problem definition for a grid world"""

    def __init__(self,
                 goal_state: Tuple[int, int], grid_width: int = 5, grid_height: int = 5,
                 init_belief=None, init_true_state: MazeState = None,
                 obs_noise: float = 0.0):
        # ----- Agent -----
        self.current_state = init_true_state
        self.policy_model = PolicyModel()
        self.transition_model = TransitionModel(grid_width, grid_height)
        self.observation_model = ObservationModel(grid_width=grid_width, grid_height=grid_height, noise=obs_noise)
        self.reward_model = RewardModel(goal_state=goal_state)

        self.agent: pomdp_py.Agent = pomdp_py.Agent(
            init_belief,
            self.policy_model,
            self.transition_model,
            self.observation_model,
            self.reward_model
        )

        # ----- Environment -----
        self.env = pomdp_py.Environment(
            init_true_state,
            self.transition_model,
            self.reward_model
        )

        # ----- Initialize POMDP -----
        super().__init__(self.agent, self.env, name="GridWorldProblem")
