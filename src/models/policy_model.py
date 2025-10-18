import random

import pomdp_py

from src.domain.action import Action


class PolicyModel(pomdp_py.RolloutPolicy):
    ACTIONS = [Action(s) for s in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]]

    def sample(self, state):
        return random.choice(self.get_all_actions())

    def rollout(self, state, history=None):
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS
