from typing import Tuple, Set

import pomdp_py

from src.models.action import Action
from src.models.maze_state import MazeState
from src.models.observation import Observation
from src.models.observation_model import ObservationModel
from src.models.reward_model import RewardModel
from src.models.transition_model import TransitionModel


class GridWorldProblem(pomdp_py.POMDP):
    """Complete POMDP problem definition for grid world"""

    def __init__(self, grid_width: int = 5, grid_height: int = 5,
                 init_state: Tuple[int, int] = (0, 0),
                 goal_states: Set[Tuple[int, int]] = None,
                 walls: Set[Tuple[int, int]] = None):

        if goal_states is None:
            goal_states = {(grid_width - 1, grid_height - 1)}

        # Create models
        transition_model = TransitionModel(grid_width, grid_height, walls)
        observation_model = ObservationModel(grid_width, grid_height, walls)
        reward_model = RewardModel(goal_states)

        # Create state space
        states = set()
        for x in range(grid_width):
            for y in range(grid_height):
                if (x, y) not in (walls if walls else set()):
                    states.add(MazeState(x, y))

        # Create action space
        actions = {
            Action(Action.UP),
            Action(Action.DOWN),
            Action(Action.LEFT),
            Action(Action.RIGHT)
        }

        # Create observation space
        observations = set()
        for x in range(grid_width):
            for y in range(grid_height):
                for wall_nearby in [True, False]:
                    observations.add(Observation(x, y, wall_nearby))

        # Initial state
        init_state_obj = MazeState(init_state[0], init_state[1])

        # Initial belief (uniform over all states)
        init_belief = pomdp_py.Histogram({state: 1.0 / len(states) for state in states})

        super().__init__(
            transition_model=transition_model,
            observation_model=observation_model,
            reward_model=reward_model,
            initial_true_state=init_state_obj,
            initial_belief=init_belief
        )
