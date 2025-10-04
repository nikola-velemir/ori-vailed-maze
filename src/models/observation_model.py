import random
from typing import Set, Tuple

import pomdp_py
from networkx.classes import neighbors

from src.domain.action import Action
from src.domain.maze_state import MazeState
from src.domain.observation import Observation


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, width: int, height: int,
                 walls: Set[Tuple[int, int]] = None,
                 noise: float = 0.9, epsilon: float = 1e-4):
        self.width = width
        self.height = height
        self.walls = walls if walls else set()
        self.noise = noise
        self.epsilon = epsilon

    def probability(self, observation: Observation, next_state: MazeState, action: Action) -> float:
        """Returns the probability of an observation given the next state."""
        if observation.x == next_state.x and observation.y == next_state.y:
            return 1 - self.noise

        neighbors = self._get_valid_neighbors(next_state)

        if (observation.x, observation.y) in neighbors:
            return  self.noise / len(neighbors) if neighbors else self.epsilon

        return self.epsilon
    def sample(self, next_state: MazeState, action: Action) -> Observation:
        """Sample an observation given the next state."""
        rand = random.random()
        if rand < 1 - self.noise:
            x,y = next_state.x, next_state.y
        else:

            neighbors = self._get_valid_neighbors(next_state)
            if neighbors:
                x, y = random.choice(neighbors)
            else:
                x, y = next_state.x, next_state.y

        return Observation(x, y)

    def _get_valid_neighbors(self, state: MazeState):
        neighbors = []
        # Only 4 cardinal directions: up, down, left, right
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = state.x + dx, state.y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.walls:
                neighbors.append((nx, ny))
        return neighbors

