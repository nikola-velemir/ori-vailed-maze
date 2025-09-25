import random

import pomdp_py

from models.observation import Observation


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, board, noise=0.15):
        self.noise = noise
        self.board = board
        self.cell_types = ['.', 'w', 'h', 't', 'g']

    def probability(self, observation, next_state, action):
        true_cell = self.board.data[next_state.agent_pos[0]][next_state.agent_pos[1]]
        if observation.cell_type == true_cell:
            return 1.0 - self.noise
        else:
            return  self.noise / (len(self.cell_types) -1)

    def sample(self, next_state, action):
        """
        Sample an observation given next_state and action
        """
        true_cell = self.board.data[next_state.agent_pos[0]][next_state.agent_pos[1]]
        if random.random() < self.noise:
            choices = [c for c in self.cell_types if c != true_cell]
            return Observation(random.choice(choices))
        else:
            return Observation(true_cell)

    def get_all_observations(self):
        """
        Enumerate all possible observations for solvers that need it
        """
        return [Observation(c) for c in self.cell_types]
