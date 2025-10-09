import pomdp_py


class Observation(pomdp_py.Observation):
    def __init__(self, x, y, north=None, south=None, east=None, west=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.y = y
        # What agent senses in each direction
        self.north = north  # 'clear', 'wall', 'trap', 'goal'
        self.south = south
        self.east = east
        self.west = west

    def __hash__(self):
        return hash((self.x, self.y, self.north, self.south, self.east, self.west))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return (self.x == other.x and
                    self.y == other.y and self.north == other.north and self.south == other.south and self.east == other.east and self.west == other.west)
        return False

    def __str__(self):
        return f'({self.x}, {self.y}, {str(self.north)}, {str(self.south)}, {str(self.east)}, {str(self.west)})'

    def __repr__(self):
        return f'Observation(({self.x}, {self.y}, {str(self.north)}, {str(self.south)}, {str(self.east)}, {str(self.west)})'