import pomdp_py

class MazeState(pomdp_py.State):
    def __init__(self, x,y):
        self.x = x
        self.y = y

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