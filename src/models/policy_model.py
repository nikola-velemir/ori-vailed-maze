import random

import pomdp_py

from models.move import Move


class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
    small, finite action space"""

    ACTIONS = [Move(s) for s in {"up", "down", "right", "left"}]


    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]


    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)


    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS
