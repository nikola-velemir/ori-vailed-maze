from typing import Tuple, Set

import pomdp_py

from src.domain.action import Action
from src.domain.maze_state import MazeState


class RewardModel(pomdp_py.RewardModel):
    def __init__(self,
                 goal_state: Tuple[int, int],
                 walls: Set[Tuple[int, int]] = None,
                 traps: Set[Tuple[int, int]] = None,
                 coins: Set[Tuple[int, int]] = None,
                 holes: Set[Tuple[int, int]] = None,
                 width: int = None,
                 height: int = None,
                 goal_reward: float = 1000.0,
                 coin_reward: float = 10.0,
                 wall_penalty: float = -10.0,
                 trap_penalty: float = -50.0,
                 hole_penalty=-100.0,
                 step_cost: float = -0.25,):

        self.hole_penalty = hole_penalty
        self.goal_state = goal_state

        self.coin_reward = coin_reward
        self.goal_reward = goal_reward
        self.step_cost = step_cost
        self.wall_penalty = wall_penalty
        self.trap_penalty = trap_penalty

        self.width = width
        self.height = height

        self.walls = walls
        self.traps = traps
        self.coins = coins
        self.holes = holes

    def _reward_func(self, state: MazeState, action: Action, next_state: MazeState) -> float:
        next_pos = (next_state.x, next_state.y)
        if next_pos == self.goal_state:
            return self.goal_reward
        if self.holes and next_pos in self.holes:
            return  self.hole_penalty
        if self.traps and next_pos in self.traps:
            return self.trap_penalty
        if next_pos in state.coins and next_pos not in next_state.coins:
            return self.coin_reward
        if next_state.x == state.x and next_state.y == state.y:
            return  self.wall_penalty
        return  self.step_cost

    def sample(self, state: MazeState, action: Action, next_state: MazeState) -> float:
        return self._reward_func(state, action, next_state)
