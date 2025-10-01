import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
import os
import sys
import time
from PIL import Image, ImageTk  # pip install --upgrade Pillow

from board import Board
from search import *
from src.board_parser.board_parser import BoardParser
from state import *

# Mapping search strategies
search_class_map = {
    "POMCP": BreadthFirstSearch,
    "POUCT": GreedySearch
}

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

class Game:
    def __init__(self, board_file = 'board.json', default_search="POMCP", cell_size=40):
        self.board_data = BoardParser.parse(board_file_path=board_file)
        self.cell_size = cell_size

        # Load board from JSON dict if provided
        if self.board_data:
            self.board = Board(self.board_data)
            self.rows = self.board.rows
            self.cols = self.board.cols
        else:
            self.rows = 5
            self.cols = 5
            self.board = Board(rows=self.rows, cols=self.cols)

        # Initialize canvas tracking
        self.grid_elem_ids = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        self.grid_text_ids = [[[] for _ in range(self.cols)] for _ in range(self.rows)]

        # Tkinter root
        self.root = tk.Tk()
        self.root.title('Vailed maze')
        self.make_menu(self.root)

        # Frames
        self.ui = tk.Frame(self.root, bg='white')
        self.ui2 = tk.Frame(self.root, bg='white')

        # Canvas
        self.canvas = tk.Canvas(self.root,
                                width=self.cols * self.cell_size + 1,
                                height=self.rows * self.cell_size + 1,
                                highlightthickness=0, bd=0, bg='white')

        # Load icons
        self.canvas.icons = dict()
        self.icons = dict()
        for f in os.listdir('icons'):
            icon = Image.open(os.path.join('icons', f))
            icon = icon.resize((self.cell_size - 2, self.cell_size - 2), Image.LANCZOS)
            self.icons[f] = ImageTk.PhotoImage(icon)

        # Search option
        self.search_class_text = tk.StringVar(self.ui)
        self.search_class_text.set(default_search)

        # Buttons and options
        search_option = tk.OptionMenu(self.ui, self.search_class_text, *search_class_map.keys())
        start_button = tk.Button(self.ui, text='SEARCH', width=10, command=self.do_search)
        restart_button = tk.Button(self.ui, text='RESET', width=10, command=self.reset)
        clear_button = tk.Button(self.ui, text='CLEAR ALL', width=10, command=self.clear)
        debug_button = tk.Button(self.ui, text='DEBUG', width=10, command=self.debug)
        stat_report = tk.Label(self.root, text='      ', bg='white', justify=tk.LEFT, relief=tk.GROOVE,
                               font=tkFont.Font(weight='bold'))

        # Grid layout
        search_option.grid(row=0, column=0, padx=10, pady=10)
        start_button.grid(row=1, column=0, padx=10, pady=10)
        clear_button.grid(row=2, column=0, padx=10, pady=10)
        restart_button.grid(row=3, column=0, padx=10, pady=10)
        debug_button.grid(row=4, column=0, padx=10, pady=10)
        stat_report.pack(side=tk.RIGHT, expand=tk.NO, fill=tk.NONE)

        # Display board
        self.display_board()
        self.canvas.bind('<Button-1>', self.switch_cell)
        self.canvas.bind('<Button-2>', self.switch_cell_backwards)
        self.ui.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)
        self.canvas.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
        self.ui2.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH, anchor=tk.W)

        self.processed = None
        self.path = None

    # ---------------- Main Loop ----------------
    def run(self):
        self.root.mainloop()

    # ---------------- Board Loading ----------------
    def load_board_from_dict(self, data_dict):
        """Load board from JSON dictionary"""
        self.board.load_from_dict(data_dict)
        self.rows = self.board.rows
        self.cols = self.board.cols
        self.grid_elem_ids = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        self.grid_text_ids = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        self.canvas.config(width=self.cols * self.cell_size + 1,
                           height=self.rows * self.cell_size + 1)
        self.display_board()

    def save_board_to_dict(self):
        """Return JSON dict representing the current board models"""
        return self.board.to_dict()

    # ---------------- Board Display ----------------
    def display_board(self):
        self.canvas.delete(tk.ALL)
        for row in range(len(self.board.data)):
            for col in range(len(self.board.data[0])):
                self.update_board(row, col)

    def update_board(self, row, col):
        data = self.board.data
        text = self.board.text
        self.delete_elems(row, col)
        elem = data[row][col]
        d, i = elem, elem
        if ',' in elem:
            s = elem.split(',')
            d = s[0]
            i = s[1]

        if d in board_to_colors:
            self.draw_rectangle(row, col, board_to_colors[d])
        if i in board_to_icons:
            icon = self.icons[board_to_icons[i]]
            self.draw_icon(row, col, icon)

        if len(text[row][col]) > 0:
            self.draw_text(row, col, text[row][col])
        else:
            self.delete_texts(row, col)

    # ---------------- Canvas Drawing ----------------
    def get_cell_rectangle(self, row, col):
        return col * self.cell_size, row * self.cell_size, (col + 1) * self.cell_size, (row + 1) * self.cell_size

    def draw_rectangle(self, row, col, color, width=1):
        rect = self.get_cell_rectangle(row, col)
        elem_id = self.canvas.create_rectangle(rect, width=width, fill=color, outline='gray')
        self.save_elem_id(elem_id, row, col)

    def draw_text(self, row, col, text):
        rect = self.get_cell_rectangle(row, col)
        l = tk.Label(self.canvas, text=text)
        l.bind('<Button-1>', lambda event: self.switch_cell(event, row, col))
        elem_id = self.canvas.create_window(rect[0] + self.cell_size/2,
                                            rect[1] + self.cell_size/2,
                                            height=self.cell_size/3,
                                            window=l)
        self.save_elem_id(elem_id, row, col)
        self.save_text_id(elem_id, row, col)

    def draw_icon(self, row, col, icon):
        rect = self.get_cell_rectangle(row, col)
        elem_id = self.canvas.create_image(rect[0]+2, rect[1]+2, image=icon, anchor=tk.NW)
        self.canvas.icons[elem_id] = icon
        self.save_elem_id(elem_id, row, col)

    # ---------------- ID Tracking ----------------
    def save_elem_id(self, elem_id, row, col):
        if len(self.grid_elem_ids[row][col]) == 0:
            self.grid_elem_ids[row][col] = []
        self.grid_elem_ids[row][col].append(elem_id)

    def save_text_id(self, elem_id, row, col):
        if len(self.grid_text_ids[row][col]) == 0:
            self.grid_text_ids[row][col] = []
        self.grid_text_ids[row][col].append(elem_id)

    def delete_elems(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            for elem_id in self.grid_elem_ids[row][col]:
                self.canvas.delete(elem_id)
                if elem_id in self.canvas.icons:
                    del self.canvas.icons[elem_id]
            self.grid_elem_ids[row][col] = []

    def delete_texts(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            for elem_id in self.grid_text_ids[row][col]:
                self.canvas.delete(elem_id)
                if elem_id in self.canvas.icons:
                    del self.canvas.icons[elem_id]
            self.grid_text_ids[row][col] = []
            self.board.text[row][col] = ''

    # ---------------- Interaction ----------------
    def switch_cell(self, event, row=None, col=None):
        if row is None and col is None:
            col = event.x // self.cell_size
            row = event.y // self.cell_size
        self.board.switch_cell(row, col)
        self.update_board(row, col)

    def switch_cell_backwards(self, event, row=None, col=None):
        if row is None and col is None:
            col = event.x // self.cell_size
            row = event.y // self.cell_size
        self.board.switch_cell_backwards(row, col)
        self.update_board(row, col)

    # ---------------- Menu ----------------
    def make_menu(self, win):
        top = tk.Menu(win)
        win.config(menu=top)
        file_menu = tk.Menu(top)
        file_menu.add_command(label='Quit', command=sys.exit)
        top.add_cascade(label='File', menu=file_menu, underline=0)

    # ---------------- Search ----------------
    def get_search_class(self):
        return search_class_map[self.search_class_text.get()]

    def do_search(self):
        self.reset()
        search_class = self.get_search_class()
        search = search_class(self.board)
        initial_state = RobotState
        start = time.perf_counter()
        path, self.processed, states = search.search(initial_state)
        end = time.perf_counter()
        self.path = list(map(lambda x: x.position, path)) if path else None

        print('-'*15, 'DONE', '-'*15)
        print('Time: {0:.4f}s'.format(end - start))
        print('Processed nodes:', len(self.processed))
        print('States left:', len(states))
        if path:
            print('Total cost:', path[-1].get_current_cost())
        else:
            print('-'*15, 'NO SOLUTION', '-'*15)

        if self.path:
            # Draw solution path
            for idx, p in enumerate(self.path):
                text = self.board.text[p[0]][p[1]]
                text = f"{text},{idx}" if text else str(idx)
                self.board.text[p[0]][p[1]] = text
                self.update_board(p[0], p[1])

    # ---------------- Debug ----------------
    def move_icon(self, from_position, to_position, has_box=None):
        f = self.board.data[from_position[0]][from_position[1]]
        t = self.board.data[to_position[0]][to_position[1]]
        if ',' in f:
            s = f.split(',')
            f = s[0]
            if has_box:
                t = 'b,' + s[1]
            else:
                t += ',' + s[1]
        else:
            t += ',' + f
            f = '.'
        self.board.data[from_position[0]][from_position[1]] = f
        self.update_board(from_position[0], from_position[1])
        self.board.data[to_position[0]][to_position[1]] = t
        self.update_board(to_position[0], to_position[1])
        self.root.update()

    def debug(self):
        self.reset()
        position = self.board.find_position('a')
        for idx, p in enumerate(self.processed):
            self.move_icon(position, p.position, hasattr(p, 'has_box') and p.has_box)
            position = p.position

    # ---------------- Clear / Reset ----------------
    def clear(self):
        self.board.clear()
        self.display_board()

    def reset(self):
        for row in range(self.rows):
            for col in range(self.cols):
                self.delete_texts(row, col)
        self.display_board()
