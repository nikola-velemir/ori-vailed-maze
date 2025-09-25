import random
from typing import Set, Tuple

import pomdp_py

from src.domain.action import Action
from src.domain.maze_state import MazeState


class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, width: int, height: int, walls: Set[Tuple[int, int]] = None, noise: float = 0.1):
        self.grid_width = width
        self.grid_height = height
        self.walls = walls if walls else set()
        self.noise = noise

    def probability(self, next_state: MazeState, state: MazeState, action: Action) -> float:
        intended_next = self._get_next_position(state, action)

        if next_state.x == intended_next.x and next_state.y == intended_next.y:
            return 1.0 - self.noise

        adjacent_positions = self._get_adjacent_positions(state)

        if (next_state.x, next_state.y) in adjacent_positions:
            return self.noise / len(adjacent_positions)

        if next_state.x == state.x and next_state.y == state.y:
            if intended_next.x == state.x and intended_next.y == state.y:
                return 1.0 - self.noise
            else:
                return self.noise / len(adjacent_positions)
        return 0.0

    def sample(self, state: MazeState, action: Action):
        if random.random() < self.noise:
            adjacent_positions = self._get_adjacent_positions(state)

            if adjacent_positions:
                x, y = random.choice(list(adjacent_positions))
                return MazeState(x, y)
            else:
                return state
        else:
            return self._get_next_position(state, action)

    def _get_adjacent_positions(self, state: MazeState) -> Set[Tuple[int, int]]:
        positions = set()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:

            x, y = state.x + dx, state.y + dy

            if 0 <= x < self.grid_width \
                    and 0 <= y < self.grid_height \
                    and (x, y) not in self.walls:
                positions.add((x, y))

        return positions

    def _get_next_position(self, state: MazeState, action: Action) -> MazeState:
        x, y = state.x, state.y
        if action.name == Action.UP:
            candidate = (x, max(0, y - 1))
        elif action.name == Action.DOWN:
            candidate = (x, min(self.grid_height - 1, y + 1))
        elif action.name == Action.LEFT:
            candidate = (max(0, x - 1), y)
        elif action.name == Action.RIGHT:
            candidate = (min(self.grid_width - 1, x + 1), y)
        else:
            candidate = (x, y)

        cx, cy = candidate

        cx = max(0, min(cx, self.grid_width - 1))
        cy = max(0, min(cy, self.grid_height - 1))

        if 0 <= cx < self.grid_width and 0 <= cy < self.grid_height and (cx, cy) not in self.walls:
            return MazeState(cx, cy)

        return MazeState(x, y)
