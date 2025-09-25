import random
import pomdp_py
from models.maze_state import MazeState  # You can rename this to GridState

class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, board, move_probabilities=None):
        """
        board: instance of your Board class
        move_probabilities: dict, e.g., {"intended": 0.8, "left_slip": 0.1, "right_slip": 0.1}
        """
        self.board = board
        self.move_probabilities = move_probabilities or {"intended": 0.8, "left_slip": 0.1, "right_slip": 0.1}
        self.directions = ["up", "down", "left", "right"]

    def probability(self, next_state, state, action):
        """
        Return P(s' | s, a)
        """
        possible_next_states = self._possible_next_states(state, action)
        total = sum(possible_next_states.values())
        return possible_next_states.get(next_state.agent_pos, 0) / total if total > 0 else 0

    def sample(self, state, action):
        """
        Sample next state according to stochastic movement
        """
        possible_next_states = self._possible_next_states(state, action)
        positions = list(possible_next_states.keys())
        probabilities = list(possible_next_states.values())
        chosen_pos = random.choices(positions, weights=probabilities, k=1)[0]
        return MazeState(agent_pos=chosen_pos)

    def _possible_next_states(self, state, action):
        """
        Compute all possible next positions given stochastic movement
        Returns dict: {position_tuple: probability}
        """
        row, col = state.agent_pos
        intended_move = self._move_delta(action.direction)
        candidates = {}

        # Intended
        r, c = row + intended_move[0], col + intended_move[1]
        if self._is_valid(r, c):
            candidates[(r, c)] = self.move_probabilities.get("intended", 0.8)
        else:
            candidates[(row, col)] = self.move_probabilities.get("intended", 0.8)

        # Left slip (turn left)
        left_dir = self._turn_left(action.direction)
        r, c = row + self._move_delta(left_dir)[0], col + self._move_delta(left_dir)[1]
        if self._is_valid(r, c):
            candidates[(r, c)] = self.move_probabilities.get("left_slip", 0.1)
        else:
            candidates[(row, col)] = self.move_probabilities.get("left_slip", 0.1)

        # Right slip (turn right)
        right_dir = self._turn_right(action.direction)
        r, c = row + self._move_delta(right_dir)[0], col + self._move_delta(right_dir)[1]
        if self._is_valid(r, c):
            candidates[(r, c)] = self.move_probabilities.get("right_slip", 0.1)
        else:
            candidates[(row, col)] = self.move_probabilities.get("right_slip", 0.1)

        return candidates

    def _is_valid(self, row, col):
        return not self.board.is_out_of_bounds(row, col) and not self.board.hits_wall(row, col)

    def _move_delta(self, direction):
        if direction == "up":
            return (-1, 0)
        elif direction == "down":
            return (1, 0)
        elif direction == "left":
            return (0, -1)
        elif direction == "right":
            return (0, 1)
        return (0, 0)

    def _turn_left(self, direction):
        turn_map = {"up": "left", "left": "down", "down": "right", "right": "up"}
        return turn_map[direction]

    def _turn_right(self, direction):
        turn_map = {"up": "right", "right": "down", "down": "left", "left": "up"}
        return turn_map[direction]

    def get_all_states(self):
        """Needed only if using solvers that enumerate states"""
        positions = [(r, c) for r in range(self.board.rows) for c in range(self.board.cols)]
        # optionally filter out walls
        positions = [pos for pos in positions if not self.board.hits_wall(pos[0], pos[1])]
        return [MazeState(agent_pos=pos) for pos in positions]
