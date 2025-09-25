import pomdp_py


class Observation(pomdp_py.Observation):
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return (self.x == other.x and
                    self.y == other.y)
        return False

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __repr__(self):
        return f'Observation({self.x}, {self.y})'
