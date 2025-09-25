import pomdp_py


class RewardModel(pomdp_py.RewardModel):
    def __init__(self, board):
        """
        board: instance of your Board class
        """
        self.board = board

    def _reward_func(self, state, action, next_state):
        r = self.board.rewards.get("step", -1)  # default step cost
        r_cell = self.board.data[next_state.agent_pos[0]][next_state.agent_pos[1]]

        if r_cell == 'g':
            r += self.board.rewards.get("goal", 100)
        elif r_cell == 'h':
            r += self.board.rewards.get("hole", -100)
        elif r_cell == 't':
            r += self.board.rewards.get("trap", -50)
        return r

    def sample(self, state, action, next_state):
        # deterministic reward
        return self._reward_func(state, action, next_state)
