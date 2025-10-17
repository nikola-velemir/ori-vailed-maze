import random
from typing import Set, Tuple

import pomdp_py

from src.domain.action import Action
from src.domain.observation import Observation
from src.domain.maze_state import MazeState


class BetterObsModel(pomdp_py.ObservationModel):
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

        # Possible sensor readings for structured noise
        self.possible_readings = ['clear', 'wall', 'something', 'coin', 'danger']

    def sample(self, next_state: MazeState, action: Action) -> Observation:
        """Sample observation - only what the agent senses"""

        sensed = {}
        for dir_name, (dx, dy) in self.directions.items():
            # Sometimes sensors fail
            if random.random() < self.sensor_failure:
                sensed[dir_name] = None  # No information
            else:
                true_content = self._sense_direction(next_state, dx, dy)

                if random.random() < self.sensor_noise:
                    # Wrong reading - use structured noise
                    sensed[dir_name] = self._get_noisy_reading(true_content)
                else:
                    # Correct reading (after mapping)
                    sensed[dir_name] = self._map_to_observation(true_content)

        return Observation(
            north=sensed['north'],
            south=sensed['south'],
            east=sensed['east'],
            west=sensed['west']
        )

    def _sense_direction(self, state: MazeState, dx: int, dy: int) -> str:
        """Get the true content of a cell in a given direction"""
        nx, ny = state.x + dx, state.y + dy

        # Out of bounds
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return 'wall'

        # Check contents (priority order matters)
        if (nx, ny) in self.walls:
            return 'wall'
        if (nx, ny) in self.holes:
            return 'hole'
        if self.traps and (nx, ny) in self.traps:
            return 'trap'
        if self.goal and (nx, ny) == self.goal:
            return 'goal'
        if (nx, ny) in state.coins:
            return 'coin'

        return 'clear'

    def _map_to_observation(self, true_content: str) -> str:
        # Traps and goals are hidden as "something"
        if true_content in ['trap', 'goal']:
            return 'something'
        # Holes are sensed as danger
        elif true_content == 'hole':
            return 'danger'
        # Everything else is reported
        else:
            return true_content

    def _get_noisy_reading(self, true_content: str) -> str:
        """Generate a noisy sensor reading with structured confusion"""
        expected = self._map_to_observation(true_content)

        # Confusion dictionary
        confusion_options = {
            'clear': ['something', 'coin'],  # Might see something that isn't there
            'wall': ['clear'],  # Might think wall is passable
            'something': ['clear', 'coin', 'danger'],  # Could misread mysterious objects
            'coin': ['clear', 'something'],  # Might miss coin or see vague something
            'danger': ['clear', 'something']  # Might not detect danger or see vague threat
        }

        options = confusion_options.get(expected, self.possible_readings)
        return random.choice(options)

    def probability(self, observation: Observation, next_state: MazeState, action: Action) -> float:
        """Calculate P(o | s', a) based on directional sensing"""
        prob = 1.0

        for dir_name, (dx, dy) in self.directions.items():
            true_content = self._sense_direction(next_state, dx, dy)
            observed_content = getattr(observation, dir_name)
            expected_content = self._map_to_observation(true_content)

            # Calculate probability for this direction
            dir_prob = self._calculate_direction_probability(
                observed_content, expected_content, true_content
            )
            prob *= dir_prob

        return max(prob, self.epsilon)

    def _calculate_direction_probability(self, observed: str, expected: str, true_content: str) -> float:
        """Calculate probability for a single direction's observation"""

        # Case 1: Sensor failed (observation is None)
        if observed is None:
            return self.sensor_failure

        # Case 2: Sensor worked
        sensor_worked_prob = 1 - self.sensor_failure

        # Case 2a: Correct observation
        if observed == expected:
            # Probability = P(sensor works) * P(correct reading | sensor works)
            return sensor_worked_prob * (1 - self.sensor_noise)

        # Case 2b: Incorrect observation (noise)
        else:
            # Check if this noise is plausible given  confusion matrix
            confusion_options = {
                'clear': ['something', 'coin'],
                'wall': ['clear'],
                'something': ['clear', 'coin', 'danger'],
                'coin': ['clear', 'something'],
                'danger': ['clear', 'something']
            }

            possible_confusions = confusion_options.get(expected, self.possible_readings)

            if observed in possible_confusions:
                # Plausible confusion
                num_confusions = len(possible_confusions)
                return sensor_worked_prob * (self.sensor_noise / num_confusions)
            else:
                # Implausible confusion - very low probability
                return self.epsilon

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