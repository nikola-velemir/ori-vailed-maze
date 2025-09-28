from __future__ import annotations

from typing import Tuple, Set

import pomdp_py
from src.domain.action import Action


class MazeState(pomdp_py.State):
    def __init__(self, x: int, y: int, width: int, height: int, coins: Set[Tuple[int, int]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.coins = coins if coins else set()

    def get_next_state(self, action: Action) -> MazeState:
        """Return the next MazeState given an action, bounded by width/height."""
        x, y = self.x, self.y

        if action.name == Action.UP:
            y = max(0, y - 1)
        elif action.name == Action.DOWN:
            y = min(self.height - 1, y + 1)
        elif action.name == Action.LEFT:
            x = max(0, x - 1)
        elif action.name == Action.RIGHT:
            x = min(self.width - 1, x + 1)

        return MazeState(x, y, self.width, self.height)

    def __hash__(self):
        return hash((self.x, self.y, self.width, self.height))

    def __eq__(self, other):
        return (
                isinstance(other, MazeState)
                and self.x == other.x
                and self.y == other.y
                and self.width == other.width
                and self.height == other.height
        )

    def __str__(self):
        return f"State({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()
