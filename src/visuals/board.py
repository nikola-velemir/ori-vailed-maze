class Board:
    """
    Class that implements a game board using a JSON-like dictionary.
    Supports agent, goal, walls, holes, traps, and POMDP rewards.
    """

    def __init__(self, data_dict=None):
        self.elems = ['.', 'w', 'a', 'g', 'h', 't']  # free, wall, agent, goal, hole, trap
        if data_dict:
            self.load_from_dict(data_dict)
        else:
            self.rows = 0
            self.cols = 0
            self.data = []
            self.text = []
            self.holes = []
            self.traps = []
            self.walls = []
            self.agent = None
            self.goal = None
            self.rewards = {"step": -1, "goal": 100}
            self.discount = 0.95
            self.observation_noise = 0.0


    def load_from_dict(self, data_dict):
        """
        Load board from a JSON dictionary.
        Expected keys: width, height, agent, goal, walls, holes, traps
        """
        self.rows = data_dict["height"]
        self.cols = data_dict["width"]
        self.data = [['.'] * self.cols for _ in range(self.rows)]
        self.text = [[''] * self.cols for _ in range(self.rows)]

        # Rewards and discount
        self.rewards = data_dict.get("rewards", {"step": -1, "goal": 100})
        self.discount = data_dict.get("discount", 0.95)
        self.observation_noise = data_dict.get("observation_noise", 0.0)
        self.walls = [tuple(pos) for pos in data_dict.get("walls", [])]

        # Place agent
        ar, ac = tuple(data_dict.get("agent", (0, 0)))
        self.data[ar][ac] = 'a'
        self.agent = (ar, ac)

        # Place goal
        goal_data = data_dict.get("goal", {"position": (self.rows-1, self.cols-1), "reward": 100, "terminal": True})
        gr, gc = tuple(goal_data["position"])
        self.data[gr][gc] = 'g'
        self.goal = (gr, gc)

        # Place walls
        for r, c in data_dict.get("walls", []):
            self.data[r][c] = 'w'

        # Place holes
        self.holes = [tuple(pos) for pos in data_dict.get("holes", {}).get("positions", [])]
        for r, c in self.holes:
            self.data[r][c] = 'h'

        # Place traps
        self.traps = [tuple(pos) for pos in data_dict.get("traps", {}).get("positions", [])]
        for r, c in self.traps:
            self.data[r][c] = 't'

    def to_dict(self):
        """
        Return a JSON-like dictionary representing the current board models.
        """
        return {
            "width": self.cols,
            "height": self.rows,
            "agent": self.find_position('a'),
            "goal": self.find_position('g'),
            "walls": self.find_all_positions('w'),
            "holes": {"positions": self.find_all_positions('h')},
            "traps": {"positions": self.find_all_positions('t')},
            "rewards": self.rewards,
            "discount": self.discount,
            "observation_noise": self.observation_noise
        }

    def switch_cell(self, row, col):
        if row < len(self.data) and col < len(self.data[0]):
            idx = self.elems.index(self.data[row][col])
            idx = (idx + 1) % len(self.elems)
            self.data[row][col] = self.elems[idx]

    def switch_cell_backwards(self, row, col):
        if row < len(self.data) and col < len(self.data[0]):
            idx = self.elems.index(self.data[row][col])
            idx = (idx - 1) % len(self.elems)
            self.data[row][col] = self.elems[idx]

    def clear(self):
        for row in range(self.rows):
            for col in range(self.cols):
                self.data[row][col] = '.'
                self.text[row][col] = ''

    def find_position(self, element):
        for row in range(self.rows):
            for col in range(self.cols):
                if self.data[row][col] == element:
                    return (row, col)
        return None

    def find_all_positions(self, element):
        positions = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.data[row][col] == element:
                    positions.append((row, col))
        return positions

    def is_out_of_bounds(self, row, col):
        return row < 0 or row >= self.rows or col < 0 or col >= self.cols

    def hits_wall(self, row, col):
        return self.data[row][col] == 'w'

    @staticmethod
    def get_direction_keyboard(direction):
        if direction == 'left':
            return 0, -1
        elif direction == 'right':
            return 0, 1
        elif direction == 'up':
            return -1, 0
        elif direction == 'down':
            return 1, 0
        else:
            return 0, 0
