import pomdp_py

from src.domain.action import Action


class MazeState(pomdp_py.State):
    def __init__(self, x, y, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y

    def get_next_state(self, action:Action):
        x, y = self.x, self.y
        if action.name == Action.UP:
            y = max(0, y - 1)
        elif action.name == Action.DOWN:
            y = min(5 - 1, y + 1)
        elif action.name == Action.LEFT:
            x = max(0, x - 1)
        elif action.name == Action.RIGHT:
            x = max(5 - 1, x + 1)

        return MazeState(x, y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, MazeState):
            return self.x == other.x and self.y == other.y
        return False

    def __str__(self):
        return f"State({self.x}, {self.y})"

    def __repr__(self):
        return self.__str__()