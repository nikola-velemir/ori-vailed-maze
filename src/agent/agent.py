import pomdp_py

from src.models.observation_model import ObservationModel
from src.models.policy_model import PolicyModel
from src.models.reward_model import RewardModel
from src.models.transition_model import TransitionModel


class MazeAgent(pomdp_py.Agent):
    def __init__(self, init_pos, init_belief, width, height, goal_pos, transition_model, reward_model,
                 observation_model, policy_model):
        self.width = width
        self.height = height
        self.goal_pos = goal_pos
        self.init_state = init_pos

        super().__init__(init_belief, policy_model, transition_model, observation_model, reward_model)
