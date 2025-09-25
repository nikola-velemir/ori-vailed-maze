from typing import Tuple, Set

import pomdp_py

from src.domain.action import Action
from src.domain.maze_state import MazeState


class RewardModel(pomdp_py.RewardModel):
    def __init__(self, goal_state: Tuple[int, int],
                 goal_reward: float = 100.0, step_cost: float = -1.0):
        self.goal_state = goal_state
        self.goal_reward = goal_reward
        self.step_cost = step_cost

    def _reward_func(self, state: MazeState, action: Action, next_state: MazeState) -> float:
        if next_state.x == self.goal_state[0] and next_state.y == self.goal_state[1]:
            return self.goal_reward
        return self.step_cost

    def sample(self, state: MazeState, action: Action, next_state: MazeState) -> float:
        return self._reward_func(state, action, next_state)

