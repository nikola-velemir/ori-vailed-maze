from typing import Tuple, Set

import pomdp_py

from src.agent.agent import MazeAgent
from src.agent.environment import MazeEnvironment
from src.domain.maze_state import MazeState
from src.models.observation_model import ObservationModel
from src.models.reward_model import RewardModel
from src.models.transition_model import TransitionModel
from src.models.policy_model import PolicyModel  # assuming you have this


class MazeProblem(pomdp_py.POMDP):
    """Complete POMDP problem definition for a grid world"""

    def __init__(self,
                 goal_state: Tuple[int, int],
                 walls: Set[Tuple[int, int]] = None,
                 holes: Set[Tuple[int, int]] = None,
                 traps: Set[Tuple[int, int]] = None,
                 coins: Set[Tuple[int, int]] = None,
                 grid_width: int = 6,
                 grid_height: int = 6,
                 init_belief=None, init_true_state: MazeState = None,
                 obs_noise: float = 0.2):
        # ----- Agent -----
        self.walls = walls if walls is not None else set()
        self.holes = holes if holes is not None else set()
        self.traps = traps if traps is not None else set()
        self.coins = coins if coins is not None else set()

        self.current_state = init_true_state
        self.policy_model = PolicyModel()
        self.transition_model = TransitionModel(grid_width, grid_height, walls=walls)
        self.observation_model = ObservationModel(width=grid_width, height=grid_height, walls=walls,
                                                  noise=obs_noise)
        self.reward_model = RewardModel(
            goal_state=goal_state,
            walls=walls,
            traps=traps,
            coins=coins,
            height=grid_height,
            width=grid_width)

        self.agent = MazeAgent(
            width=grid_width,
            height=grid_height,
            init_belief=init_belief,
            goal_pos=goal_state,
            init_pos=init_true_state,
            policy_model=self.policy_model,
            transition_model=self.transition_model,
            reward_model=self.reward_model,
            observation_model=self.observation_model,
        )
        self.env = MazeEnvironment(
            width=grid_width,
            walls=self.walls,
            holes=self.holes,
            noise=obs_noise,
            init_state=init_true_state,
            goal_state=goal_state,
            reward_model=self.reward_model,
            observation_model=self.observation_model,
            transition_model=self.transition_model,
            height=grid_height,
            traps=set()
        )
        super().__init__(self.agent, self.env, name="GridWorldProblem")
