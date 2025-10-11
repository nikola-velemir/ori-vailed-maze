import pomdp_py


class Observation(pomdp_py.Observation):
    def __init__(self, north=None, south=None, east=None, west=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # What agent senses in each direction
        self.north = north  # 'clear', 'wall', 'trap', 'goal'
        self.south = south
        self.east = east
        self.west = west

    def __hash__(self):
        return hash((self.north, self.south, self.east, self.west))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return \
                    self.north == other.north and \
                    self.south == other.south and \
                    self.east == other.east and \
                    self.west == other.west
        return False

    def __str__(self):
        return f'(N:{str(self.north)}, S:{str(self.south)}, E:{str(self.east)}, W:{str(self.west)})'

    def __repr__(self):
        return f'Observation(N:{self.north}, S:{self.south}, E:{self.east}, W:{self.west})'
