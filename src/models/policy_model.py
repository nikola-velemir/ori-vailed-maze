import random

import pomdp_py

from src.domain.action import Action


class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
    small, finite action space"""

    ACTIONS = [Action(s) for s in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]]

    def sample(self, state):
        return random.choice(self.get_all_actions())  # simpler than random.sample

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS
