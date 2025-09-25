import random

import pomdp_py

from src.domain.action import Action
from src.domain.maze_state import MazeState
from src.domain.observation import Observation


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, grid_width: int, grid_height: int,
                noise: float = 0.2):

        self.grid_width = grid_width
        self.grid_height = grid_height


        self.noise = noise

    def probability(self, observation: Observation, next_state: MazeState, action: Action) -> float:
        true_obs = Observation(next_state.x, next_state.y)
        if observation == true_obs:
            return  1- self.noise

        dx = abs(observation.x - next_state.x)
        dy = abs(observation.y - next_state.y)

        if dx<= 1 and dy <= 1 \
            and 0<= observation.x < self.grid_width \
                and 0<= next_state.y < self.grid_height:
            num_neighbors = min(9, self.grid_height* self.grid_width)-1
            return self.noise / num_neighbors
        return 1e-5

    def sample(self, next_state: MazeState, action: Action) -> Observation:
        if random.random() < self.noise:
            x = max(0, min(self.grid_width - 1, next_state.x + random.randint(-1, 1)))
            y = max(0, min(self.grid_height - 1, next_state.y + random.randint(-1, 1)))

        else:
            x, y = next_state.x, next_state.y

        return Observation(x, y)

