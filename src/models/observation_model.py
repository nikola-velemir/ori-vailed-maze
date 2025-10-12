import random
from typing import Set, Tuple

import pomdp_py

from src.domain.action import Action
from src.domain.observation import Observation
from src.domain.maze_state import MazeState


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, width: int, height: int,
                 walls: Set[Tuple[int, int]] = None,
                 traps: Set[Tuple[int, int]] = None,
                 coins: Set[Tuple[int, int]] = None,
                 holes: Set[Tuple[int, int]] = None,
                 goal: Tuple[int, int] = None,
                 sensor_noise: float = 0.4,  # Sensor errors
                 sensor_failure: float = 0.1,  # Complete sensor failure
                 epsilon: float = 1e-3):
        self.width = width
        self.height = height

        self.holes = holes if holes else set()
        self.walls = walls if walls else set()
        self.traps = traps if traps else set()
        self.coins = coins if coins else set()

        self.goal = goal
        self.sensor_noise = sensor_noise
        self.sensor_failure = sensor_failure
        self.epsilon = epsilon

        self.directions = {
            'north': (0, -1),
            'south': (0, 1),
            'east': (1, 0),
            'west': (-1, 0)
        }

    def sample(self, next_state: MazeState, action: Action) -> Observation:
        """Sample observation - only what the agent senses, no position info"""

        sensed = {}
        for dir_name, (dx, dy) in self.directions.items():
            # Sometimes sensors fail
            if random.random() < self.sensor_failure:
                sensed[dir_name] = None  # No information
            else:
                true_content = self._sense_direction(next_state, dx, dy)

                if random.random() < self.sensor_noise:
                    # Wrong reading
                    sensed[dir_name] = random.choice(['clear', 'wall', 'something', 'coin', 'danger'])
                else:
                    # Generic "something"
                    if true_content in ['trap', 'goal']:
                        sensed[dir_name] = 'something'
                    # Sense coins
                    elif true_content == 'coin':
                        sensed[dir_name] = 'coin'
                    # Sense holes
                    elif true_content == 'hole':
                        sensed[dir_name] = 'danger'
                    else:
                        sensed[dir_name] = true_content

        return Observation(
            north=sensed['north'],
            south=sensed['south'],
            east=sensed['east'],
            west=sensed['west']
        )

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
        if (nx, ny) in state.coins:
            return 'coin'
        if (nx, ny) in self.holes:
            return 'hole'

        return 'clear'

    def probability(self, observation: Observation, next_state: MazeState, action: Action) -> float:
        """Calculate P(o | s', a) - only based on directional sensing"""
        prob = 1.0

        # Only directional sensing probability matters now
        for dir_name, (dx, dy) in self.directions.items():
            true_content = self._sense_direction(next_state, dx, dy)
            observed_content = getattr(observation, dir_name)

            # Sensor failed, any observation is possible
            if observed_content is None:
                prob *= self.sensor_failure
            else:
                prob *= (1 - self.sensor_failure)


                if true_content in ['trap', 'goal']:
                    expected = 'something'
                elif true_content == 'coin':
                    expected = 'coin'
                elif true_content == 'hole' or true_content == 'danger':
                    expected = 'danger'
                else:
                    expected = true_content

                if observed_content == expected:
                    prob *= (1 - self.sensor_noise)
                else:
                    prob *= self.sensor_noise / 5

        return max(prob, self.epsilon)

    def get_all_observations(self) -> list:
        """Generate all possible observation combinations"""
        observations = []
        possible_values = ['clear', 'wall', 'something', 'coin', 'danger', None]

        for north in possible_values:
            for south in possible_values:
                for east in possible_values:
                    for west in possible_values:
                        observations.append(Observation(
                            north=north, south=south,
                            east=east, west=west
                        ))

        return observations
