import pomdp_py


class Action(pomdp_py.Action):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    def __init__(self, name: str):
        if name not in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
            raise ValueError(f"Invalid action: {name}")
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Action(%s)" % self.name
