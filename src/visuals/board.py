import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
import os
import sys
import time
from PIL import Image, ImageTk

# Mapping board symbols to colors and icons
board_to_colors = {
    '.': 'white',
    'w': 'darkgray',
}

board_to_icons = {
    'a': 'agent.png',
    'g': 'goal.png',
    'h': 'hole.png',
    't': 'trap.jpg'
}


class Board:
    """
    Class that implements a game board using a JSON-like dictionary.
    Supports agent, goal, walls, holes, traps, and POMDP rewards.
    NOTE: Board uses (y, x) indexing where y=row, x=col
    """

    def __init__(self, data_dict=None, rows=None, cols=None):
        self.elems = ['.', 'w', 'a', 'g', 'h', 't']

        if data_dict:
            self.load_from_dict(data_dict)
        elif rows and cols:
            self.rows = rows
            self.cols = cols
            self.data = [['.'] * self.cols for _ in range(self.rows)]
            self.text = [[''] * self.cols for _ in range(self.rows)]
            self.holes = []
            self.traps = []
            self.walls = []
            self.agent = (0, 0)  # (y, x)
            self.goal = (rows - 1, cols - 1)  # (y, x)
            self.rewards = {"step": -1, "goal": 100}
            self.discount = 0.95
            self.observation_noise = 0.0
            self.data[0][0] = 'a'
            self.data[rows - 1][cols - 1] = 'g'
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
        """Load board from a JSON dictionary."""
        self.rows = data_dict["height"]
        self.cols = data_dict["width"]
        self.data = [['.'] * self.cols for _ in range(self.rows)]
        self.text = [[''] * self.cols for _ in range(self.rows)]

        self.rewards = data_dict.get("rewards", {"step": -1, "goal": 100})
        self.discount = data_dict.get("discount", 0.95)
        self.observation_noise = data_dict.get("observation_noise", 0.0)
        wall_dict = data_dict.get("walls", {})
        self.walls = [tuple(pos[::-1]) for pos in wall_dict.get("positions", [])]

        # Place agent - (y, x)
        agent_data = data_dict.get("agent", (0, 0))
        self.agent = tuple(agent_data) if isinstance(agent_data, (list, tuple)) else (0, 0)
        ax, ay = self.agent
        self.data[ay][ax] = 'a'

        # Place goal - (y, x)
        goal_data = data_dict.get("goal", (self.rows - 1, self.cols - 1))
        if isinstance(goal_data, dict):
            self.goal = tuple(goal_data["position"])
        else:
            self.goal = tuple(goal_data)
        gx, gy = self.goal
        self.data[gy][gx] = 'g'

        # Place walls - (y, x)
        for x, y in self.walls:
            self.data[x][y] = 'w'

        # Place holes - (y, x)
        holes_data = data_dict.get("holes", [])
        if isinstance(holes_data, dict):
            self.holes = [tuple(pos[::-1]) for pos in holes_data.get("positions", [])]
        else:
            self.holes = [tuple(pos[::-1]) for pos in holes_data]
        for y, x in self.holes:
            self.data[y][x] = 'h'

        # Place traps - (y, x)
        traps_data = data_dict.get("traps", [])
        if isinstance(traps_data, dict):
            self.traps = [tuple(pos[::-1]) for pos in traps_data.get("positions", [])]
        else:
            self.traps = [tuple(pos[::-1]) for pos in traps_data]
        for y, x in self.traps:
            self.data[y][x] = 't'

    def to_dict(self):
        """Return a JSON-like dictionary representing the current board."""
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
        """Find first occurrence of element. Returns (y, x) where y=row, x=col."""
        for x in range(self.rows):
            for y in range(self.cols):
                cell = self.data[y][x]
                if ',' in cell:
                    cell = cell.split(',')[1]
                if cell == element:
                    return (y, x)
        return None

    def find_all_positions(self, element):
        """Find all occurrences of element. Returns list of (y, x) where y=row, x=col."""
        positions = []
        for y in range(self.rows):
            for x in range(self.cols):
                if self.data[y][x] == element:
                    positions.append((y, x))
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
