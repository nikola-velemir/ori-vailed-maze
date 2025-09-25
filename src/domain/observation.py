import pomdp_py


class Observation(pomdp_py.Observation):
    def __init__(self, x:int, y:int, wall:bool = True):
        self.x = x
        self.y = y
        self.wall = wall

    def __hash__(self):
        return hash((self.x, self.y, self.wall_nearby))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return (self.x == other.x and
                    self.y == other.y and
                    self.wall == other.wall)
        return False

    def __str__(self):
        return self.__str__()

    def __repr__(self):
        return f'Observation({self.x}, {self.y}, {self.wall})'