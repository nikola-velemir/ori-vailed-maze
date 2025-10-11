import random
from typing import Set, Tuple

import pomdp_py
from sympy.physics.paulialgebra import epsilon

from src.domain.action import Action
from src.domain.maze_state import MazeState


class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self,
                 width: int, height: int,
                 walls: Set[Tuple[int, int]] = None,
                 coins: Set[Tuple[int, int]] = None,
                 move_probabilities: dict = None,  # Changed from 'noise'
                 epsilon: float = 1e-4):
        self.width = width
        self.height = height
        self.walls = walls if walls else set()
        self.coins = coins if coins else set()

        # Default to your JSON values
        self.move_probs = move_probabilities or {
            'intended': 0.8,
            'left_slip': 0.1,
            'right_slip': 0.1
        }
        self.epsilon = epsilon

        # Define perpendicular directions for slipping
        self.perpendiculars = {
            Action.UP: (Action.LEFT, Action.RIGHT),  # left, right
            Action.DOWN: (Action.RIGHT, Action.LEFT),
            Action.RIGHT: (Action.UP, Action.DOWN),
            Action.LEFT: (Action.DOWN, Action.UP)
        }

    def probability(self, next_state: MazeState, state: MazeState, action: Action) -> float:
        """Returns P(s' | s, a)"""
        # Get all three possible outcomes
        outcomes = self._get_action_outcomes(state, action)

        # Check if next_state matches any outcome
        for outcome_state, prob in outcomes:
            if (next_state.x == outcome_state.x and
                    next_state.y == outcome_state.y and
                    next_state.coins == outcome_state.coins):
                return prob

        return self.epsilon

    def sample(self, state: MazeState, action: Action) -> MazeState:
        """Sample next state according to transition probabilities"""
        outcomes = self._get_action_outcomes(state, action)

        states = [s for s, _ in outcomes]
        probs = [p for _, p in outcomes]

        # Sample according to probabilities
        return random.choices(states, weights=probs)[0]

    def _get_action_outcomes(self, state: MazeState, action: Action):
        """
        Returns list of (MazeState, probability) tuples for the three
        possible outcomes: intended direction, left slip, right slip
        """
        outcomes = []

        # 1. Intended direction (80%)
        intended_state = self._apply_action(state, action)
        outcomes.append((intended_state, self.move_probs['intended']))

        # 2. Left slip (10%)
        left_action, right_action = self.perpendiculars[action.name]
        left_state = self._apply_action(state, Action(left_action))
        outcomes.append((left_state, self.move_probs['left_slip']))

        # 3. Right slip (10%)
        right_state = self._apply_action(state, Action(right_action))
        outcomes.append((right_state, self.move_probs['right_slip']))

        return outcomes

    def _apply_action(self, state: MazeState, action: Action) -> MazeState:
        """
        Apply action to state. If move is blocked (wall/boundary),
        agent stays in place.
        """
        x, y = state.x, state.y

        if action.name == Action.UP:
            candidate = (x, y - 1)
        elif action.name == Action.DOWN:
            candidate = (x, y + 1)
        elif action.name == Action.LEFT:
            candidate = (x - 1, y)
        elif action.name == Action.RIGHT:
            candidate = (x + 1, y)
        else:
            candidate = (x, y)

        cx, cy = candidate

        # Check if move is valid
        if (0 <= cx < self.width and
                0 <= cy < self.height and
                (cx, cy) not in self.walls):
            # Valid move
            new_coins = set(state.coins)
            if (cx, cy) in new_coins:
                new_coins.remove((cx, cy))
            return MazeState(cx, cy, height=self.height, width=self.width, coins=new_coins)
        else:
            # Blocked - stay in place
            return MazeState(x, y, height=self.height, width=self.width, coins=state.coins)
