import random
from typing import Tuple, Set

import pomdp_py

from src.domain.action import Action
from src.domain.maze_state import MazeState
from src.domain.observation import Observation


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self,
                 grid_width: int,
                 grid_height: int,
                 walls: Set[Tuple[int, int]] = None,
                 noise: float = 0.2
                 ):

        self.grid_width = grid_width
        self.grid_height = grid_height

        self.walls = walls if walls else set()
        self.noise = noise


    def probability(self, observation: Observation, next_state: MazeState, action: Action) -> float:
        true_obs = Observation(next_state.x, next_state.y)
        if observation == true_obs:
            return 1 - self.noise

        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = next_state.x + dx, next_state.y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    if (nx, ny) not in self.walls and (nx, ny) != (next_state.x, next_state.y):
                        neighbors.append((nx, ny))

        if (observation.x, observation.y) in neighbors:
            return self.noise / len(neighbors) if neighbors else 0.0

        return 1e-5

    def sample(self, next_state: MazeState, action: Action) -> Observation:
        if random.random() < self.noise:
            while True:
                x = max(0, min(self.grid_width - 1, next_state.x + random.randint(-1, 1)))
                y = max(0, min(self.grid_height - 1, next_state.y + random.randint(-1, 1)))
                if (x, y) not in self.walls:
                    break
        else:
            x, y = next_state.x, next_state.y
        return Observation(x, y)
