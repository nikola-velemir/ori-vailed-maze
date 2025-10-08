import random
from typing import Set, Tuple

import pomdp_py

from src.domain.action import Action
from src.domain.better_observation import BetterObservation
from src.domain.maze_state import MazeState


class BetterObservationModel(pomdp_py.ObservationModel):
    def __init__(self, width: int, height: int,
                 walls: Set[Tuple[int, int]] = None,
                 traps: Set[Tuple[int, int]] = None,
                 goal: Tuple[int, int] = None,
                 position_noise: float = 0.2,  # noise in x,y position
                 sensor_noise: float = 0.1,  # noise in directional sensing
                 epsilon: float = 1e-3):
        self.width = width
        self.height = height
        self.walls = walls if walls else set()
        self.traps = traps if traps else set()
        self.goal = goal
        self.position_noise = position_noise
        self.sensor_noise = sensor_noise
        self.epsilon = epsilon

        self.directions = {
            'north': (0, -1),
            'south': (0, 1),
            'east': (1, 0),
            'west': (-1, 0)
        }

    def sample(self, next_state: MazeState, action: Action) -> BetterObservation:
        """Sample observation with both position and directional sensing"""

        if random.random() < self.position_noise:
            neighbors = self._get_valid_neighbors(next_state)
            if neighbors:
                x, y = random.choice(neighbors)
            else:
                x, y = next_state.x, next_state.y
        else:
            x, y = next_state.x, next_state.y

        sensed = {}
        for dir_name, (dx, dy) in self.directions.items():
            true_content = self._sense_direction(next_state, dx, dy)

            if random.random() < self.sensor_noise:
                sensed[dir_name] = random.choice(['clear', 'wall', 'trap'])
            else:
                sensed[dir_name] = true_content

        return BetterObservation(x, y,
                           north=sensed['north'],
                           south=sensed['south'],
                           east=sensed['east'],
                           west=sensed['west'])

    def _sense_direction(self, state: MazeState, dx: int, dy: int) -> str:
        """What's actually in this direction?"""
        nx, ny = state.x + dx, state.y + dy

        # Out of bounds
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return 'wall'

        # Check contents
        if (nx, ny) in self.walls:
            return 'wall'
        if self.traps and (nx, ny) in self.traps:
            return 'trap'
        if self.goal and (nx, ny) == self.goal:
            return 'goal'

        return 'clear'

    def probability(self, observation: BetterObservation, next_state: MazeState, action: Action) -> float:
        """Calculate P(o | s', a)"""
        prob = 1.0

        # 1. Position probability (as before)
        if observation.x == next_state.x and observation.y == next_state.y:
            prob *= (1 - self.position_noise)
        else:
            neighbors = self._get_valid_neighbors(next_state)
            if (observation.x, observation.y) in neighbors:
                prob *= self.position_noise / len(neighbors) if neighbors else self.epsilon
            else:
                return self.epsilon

        # 2. Directional sensing probability
        for dir_name, (dx, dy) in self.directions.items():
            true_content = self._sense_direction(next_state, dx, dy)
            observed_content = getattr(observation, dir_name)

            if observed_content == true_content:
                prob *= (1 - self.sensor_noise)
            else:
                prob *= self.sensor_noise / 2  # Could be wrong in 2 ways

        return max(prob, self.epsilon)

    def _get_valid_neighbors(self, state: MazeState):
        neighbors = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = state.x + dx, state.y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.walls:
                neighbors.append((nx, ny))
        return neighbors