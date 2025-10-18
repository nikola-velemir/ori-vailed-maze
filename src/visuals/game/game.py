import os
import sys
import tkinter as tk
import tkinter.font as tkFont
from copy import deepcopy
from tkinter import filedialog

from PIL import Image, ImageTk  # pip install --upgrade Pillow

from src.board_parser.board_parser import BoardParser
from src.visuals.game.board import Board
from src.visuals.runner import run_pomdp_simulation

# Mapping board symbols to colors and icons
board_to_colors = {
    '.': 'white',
    'w': 'darkgray',
}

board_to_icons = {
    'a': 'agent.png',
    'g': 'goal.jpg',
    'h': 'hole.png',
    't': 'trap.jpg',
    'c': 'small_reward.jpg'
}


class Game:
    def __init__(self, board_file='board.json', cell_size=40):
        self.board_data = BoardParser.parse(board_file_path=board_file)
        self.solver_config = self.board_data['solver_config']
        self.gamma = self.solver_config['discount_factor']

        self.board_file = board_file
        self.cell_size = cell_size
        self.original_board = deepcopy(self.board_data)
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

        # Move history data
        self.move_history = []

        # Canvas
        self.canvas = tk.Canvas(self.root,
                                width=self.cols * self.cell_size + 1,
                                height=self.rows * self.cell_size + 1,
                                highlightthickness=0, bd=0, bg='white')

        # Load icons
        self.canvas.icons = dict()
        self.icons = dict()
        for f in os.listdir('visuals/icons'):
            icon = Image.open(os.path.join('visuals/icons', f))
            icon = icon.resize((self.cell_size - 2, self.cell_size - 2), Image.LANCZOS)
            self.icons[f] = ImageTk.PhotoImage(icon)

        self.search_class_text = tk.StringVar(self.ui)
        self.search_class_text.set("POUCT")  # default value

        planner_options = ["POUCT", "POMCP"]
        planner_dropdown = tk.OptionMenu(self.ui, self.search_class_text, *planner_options)
        planner_dropdown.config(width=10)
        planner_dropdown.grid(row=1, column=0, padx=10, pady=10)

        restart_button = tk.Button(self.ui, text='RESET', width=10, command=self.reset)

        pouct_button = tk.Button(self.ui, text='RUN POMDP', width=10, command=self.run_simulation)
        pouct_button.grid(row=5, column=0, padx=10, pady=10)

        restart_button.grid(row=3, column=0, padx=10, pady=10)

        self.show_heatmap_var = tk.BooleanVar(value=True)
        heatmap_checkbox = tk.Checkbutton(
            self.ui,
            text="Show Heatmap",
            variable=self.show_heatmap_var,
            onvalue=True,
            offvalue=False,
            bg='white'
        )
        heatmap_checkbox.grid(row=6, column=0, padx=10, pady=5)

        # Move history frame with scrollbar - using grid
        self.history_frame = tk.Frame(self.ui, bg='white', relief=tk.GROOVE, borderwidth=2)
        self.history_frame.grid(row=7, column=0, padx=10, pady=10, sticky='nsew')

        self.history_label = tk.Label(self.history_frame, text="Move History", bg='white',
                                      font=tkFont.Font(weight='bold'))
        self.history_label.pack(side=tk.TOP, pady=5)

        # Create canvas and scrollbar for history
        self.history_canvas = tk.Canvas(self.history_frame, width=230, height=300, bg='white', highlightthickness=0)
        self.history_scrollbar = tk.Scrollbar(self.history_frame, orient="vertical", command=self.history_canvas.yview)
        self.history_scrollable_frame = tk.Frame(self.history_canvas, bg='white')

        self.history_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))
        )

        self.history_canvas.create_window((0, 0), window=self.history_scrollable_frame, anchor="nw")
        self.history_canvas.configure(yscrollcommand=self.history_scrollbar.set)

        self.history_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Display board
        self.display_board()

        stat_report = tk.Label(self.root, text='      ', bg='white', justify=tk.LEFT, relief=tk.GROOVE,
                               font=tkFont.Font(weight='bold'))

        self.ui.grid(row=0, column=1, sticky='ns', padx=5, pady=5)
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.ui2.grid(row=1, column=0, columnspan=2, sticky='ew')
        stat_report.grid(row=2, column=0, columnspan=2, sticky='ew')

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
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

    def open_file(self):
        """Open a file dialog to select a board JSON file"""
        filename = filedialog.askopenfilename(
            title="Select a board file",
            filetypes=(
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ),
            initialdir="."  # Start in current directory
        )

        if filename:  # If user selected a file (didn't cancel)
            try:
                # Parse the new board file
                self.board_file = filename
                self.board_data = BoardParser.parse(board_file_path=filename)
                self.solver_config = self.board_data['solver_config']
                self.gamma = self.solver_config['discount_factor']
                self.original_board = deepcopy(self.board_data)

                # Reload the board
                self.load_board_from_dict(self.board_data)
                print(f"✓ Successfully loaded: {filename}")

            except Exception as e:
                print(f"❌ Error loading file: {e}")
                # Optionally show an error dialog
                from tkinter import messagebox
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

    # Update the make_menu method:
    def make_menu(self, win):
        top = tk.Menu(win)
        win.config(menu=top)
        file_menu = tk.Menu(top)

        # Add "Open" option before "Quit"
        file_menu.add_command(label='Open...', command=self.open_file, accelerator='Ctrl+O')
        file_menu.add_separator()  # Add a visual separator
        file_menu.add_command(label='Quit', command=sys.exit)

        top.add_cascade(label='File', menu=file_menu, underline=0)

        top.add_command(label='Reload Board', command=self.reload_board)

        # Optional: Bind keyboard shortcut
        win.bind('<Control-o>', lambda e: self.open_file())

    # ---------------- Board Display ----------------
    def reload_board(self):
        """Reload the last loaded board file."""
        try:
            self.board_data = BoardParser.parse(board_file_path=self.board_file)
            self.solver_config = self.board_data['solver_config']
            self.gamma = self.solver_config['discount_factor']
            self.original_board = deepcopy(self.board_data)
            self.load_board_from_dict(self.board_data)
            print(f"✓ Reloaded board from: {self.board_file}")
        except Exception as e:
            print(f"❌ Error reloading file: {e}")
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to reload file:\n{str(e)}")

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
        elem_id = self.canvas.create_window(rect[0] + self.cell_size / 2,
                                            rect[1] + self.cell_size / 2,
                                            height=self.cell_size / 3,
                                            window=l)
        self.save_elem_id(elem_id, row, col)
        self.save_text_id(elem_id, row, col)

    def draw_icon(self, row, col, icon):
        rect = self.get_cell_rectangle(row, col)
        elem_id = self.canvas.create_image(rect[0] + 2, rect[1] + 2, image=icon, anchor=tk.NW)
        self.canvas.icons[elem_id] = icon
        self.save_elem_id(elem_id, row, col)


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

    def move_icon_xy(self, from_xy, to_xy):
        self.move_icon((from_xy[1], from_xy[0]), (to_xy[1], to_xy[0]))

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


    def reset(self):
        """Resets the board"""
        self.load_board_from_dict(self.original_board)
        for row in range(self.rows):
            for col in range(self.cols):
                self.delete_texts(row, col)
        self.display_board()
        self.clear_move_history()

    def run_simulation(self):
        """Runs the simulation in friend function"""
        run_pomdp_simulation(self)

    def print_console_grid(self, agent_state, goal_state, walls, traps, coins):
        """Prints a simple ASCII grid showing agent, goal, and obstacles."""
        print()
        for y in range(self.rows):
            row = ""
            for x in range(self.cols):
                if agent_state.x == x and agent_state.y == y:
                    row += "A "
                elif goal_state and goal_state[0] == x and goal_state[1] == y:
                    row += "G "
                elif (x, y) in walls:
                    row += "# "
                elif (x, y) in traps:
                    row += "T "
                elif (x, y) in coins:
                    row += "C "
                else:
                    row += ". "
            print(row)
        print()

    def get_rewards(self):
        """Extracts rewards from board data."""
        data = self.board_data
        hole_penalty = data['holes']['penalty']
        trap_penalty = data['traps']['penalty']
        step_cost = data['rewards']['step']
        wall_penalty = data['walls']['penalty']
        goal_reward = data['goal']['reward']
        coins_reward = data['coins']['reward']
        return {
            'hole_penalty': hole_penalty,
            'trap_penalty': trap_penalty,
            'step_cost': step_cost,
            'wall_penalty': wall_penalty,
            'goal_reward': goal_reward,
            'coin_reward': coins_reward,
        }

    def get_observation_noise(self):
        return self.board_data['observation_noise']

    def get_move_probabilites(self):
        return self.board_data['move_probabilities']

    def add_move_to_history(self, action, start_pos, end_pos, reward):
        """Adds a move record to the history display."""
        move_num = len(self.move_history) + 1
        self.move_history.append({
            'action': action,
            'start': start_pos,
            'end': end_pos,
            'reward': reward
        })

        # Create move entry
        move_frame = tk.Frame(self.history_scrollable_frame, bg='lightgray', relief=tk.RAISED, borderwidth=1)
        move_frame.pack(fill=tk.X, padx=5, pady=3)

        # Move number
        tk.Label(move_frame, text=f"Move {move_num}", bg='lightgray', font=tkFont.Font(weight='bold')).pack(anchor=tk.W,
                                                                                                            padx=5,
                                                                                                            pady=2)

        # Action
        tk.Label(move_frame, text=f"Action: {action}", bg='lightgray').pack(anchor=tk.W, padx=5)

        # Positions
        tk.Label(move_frame, text=f"From: ({start_pos[0]}, {start_pos[1]})", bg='lightgray').pack(anchor=tk.W, padx=5)
        tk.Label(move_frame, text=f"To: ({end_pos[0]}, {end_pos[1]})", bg='lightgray').pack(anchor=tk.W, padx=5)

        # Reward with color coding
        reward_color = 'green' if reward > 0 else 'red' if reward < 0 else 'black'
        reward_label = tk.Label(move_frame, text=f"Reward: {reward:.2f}", bg='lightgray', fg=reward_color,
                                font=tkFont.Font(weight='bold'))
        reward_label.pack(anchor=tk.W, padx=5, pady=2)

        # Auto-scroll to bottom
        self.history_canvas.update_idletasks()
        self.history_canvas.yview_moveto(1.0)

    def clear_move_history(self):
        """Clears all move history entries."""
        self.move_history = []
        for widget in self.history_scrollable_frame.winfo_children():
            widget.destroy()