import os
import sys
import time
import tkinter as tk
import tkinter.font as tkFont
from copy import deepcopy
from tkinter import filedialog

import matplotlib.pyplot as plt
from PIL import Image, ImageTk  # pip install --upgrade Pillow

from src.agent.belief import initialize_uniform_belief
from src.board_parser.board_parser import BoardParser
from src.utils.metric_utils import calculate_discounted_reward, calculate_total_sum_reward
from src.visuals.game.board import Board
from src.visuals.heatmap.heatmap_manager import HeatmapManager
from src.visuals.heatmap.heatmap_window import TkinterHeatmapWindow

# Mapping board symbols to colors and icons
board_to_colors = {
    '.': 'white',
    'w': 'darkgray',
}

board_to_icons = {
    'a': 'agent.png',
    'g': 'goal.jpg',
    'h': 'hole.png',
    't': 'trap.jpg'
}


class Game:
    def __init__(self, board_file='board.json', default_search="POMCP", cell_size=40):
        self.board_data = BoardParser.parse(board_file_path=board_file)
        self.solver_config = self.board_data['solver_config']
        self.gamma = self.solver_config['discount_factor']
        print(self.board_data)
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
        debug_button = tk.Button(self.ui, text='DEBUG', width=10, command=self.debug)
        stat_report = tk.Label(self.root, text='      ', bg='white', justify=tk.LEFT, relief=tk.GROOVE,
                               font=tkFont.Font(weight='bold'))
        pouct_button = tk.Button(self.ui, text='RUN POMDP', width=10, command=self.run_pomdp_simulation)
        pouct_button.grid(row=5, column=0, padx=10, pady=10)

        restart_button.grid(row=3, column=0, padx=10, pady=10)
        debug_button.grid(row=4, column=0, padx=10, pady=10)
        stat_report.pack(side=tk.RIGHT, expand=tk.NO, fill=tk.NONE)

        # Display board
        self.display_board()
        self.ui.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)
        self.canvas.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
        self.ui2.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH, anchor=tk.W)
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
                self.board_data = BoardParser.parse(board_file_path=filename)
                self.solver_config = self.board_data['solver_config']
                self.gamma = self.solver_config['discount_factor']
                self.original_board = deepcopy(self.board_data)

                # Reload the board
                self.load_board_from_dict(self.board_data)
                print(f"✓ Successfully loaded: {filename}")
                print(self.board_data)

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

        # Optional: Bind keyboard shortcut
        win.bind('<Control-o>', lambda e: self.open_file())
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


    #
    # def do_search(self):
    #     self.reset()
    #     search_class = self.get_search_class()
    #     search = search_class(self.board)
    #     initial_state = RobotState
    #     start = time.perf_counter()
    #     path, self.processed, states = search.search(initial_state)
    #     end = time.perf_counter()
    #     self.path = list(map(lambda x: x.position, path)) if path else None
    #
    #     print('-' * 15, 'DONE', '-' * 15)
    #     print('Time: {0:.4f}s'.format(end - start))
    #     print('Processed nodes:', len(self.processed))
    #     print('States left:', len(states))
    #     if path:
    #         print('Total cost:', path[-1].get_current_cost())
    #     else:
    #         print('-' * 15, 'NO SOLUTION', '-' * 15)
    #
    #     if self.path:
    #         # Draw solution path
    #         for idx, p in enumerate(self.path):
    #             text = self.board.text[p[0]][p[1]]
    #             text = f"{text},{idx}" if text else str(idx)
    #             self.board.text[p[0]][p[1]] = text
    #             self.update_board(p[0], p[1])

    def move_icon_xy(self, from_xy, to_xy):
        self.move_icon((from_xy[1], from_xy[0]), (to_xy[1], to_xy[0]))

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
        self.load_board_from_dict(self.original_board)
        for row in range(self.rows):
            for col in range(self.cols):
                self.delete_texts(row, col)
        self.display_board()

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

    def run_pomdp_simulation(self):
        rewards = []
        planner_name = self.search_class_text.get()
        self.reset()
        """Run a POMDP simulation and visually update the board after each step."""
        from src.domain.action import Action
        from src.domain.maze_state import MazeState
        from src.problem.problem import MazeProblem
        from src.visuals.heatmap.heatmap_utils import show_histogram

        print("\n▶ Starting POMDP simulation...")

        # --- Extract board setup ---
        grid_height = self.rows
        grid_width = self.cols
        r, c = self.board.find_position('g')
        goal_state = (c, r)
        c, r = self.board.find_position('a')
        start_state = (c, r)

        if not goal_state or not start_state:
            print("❌ Missing agent or goal position on the board.")
            return

        walls = set((c, r) for r, c in self.board.find_all_positions('w'))
        holes = set((c, r) for r, c in self.board.find_all_positions('h'))
        traps = set((c, r) for r, c in self.board.find_all_positions('t'))
        coins = set()

        # --- Initialize belief + state ---
        init_belief_state = initialize_uniform_belief(planner_name, grid_width, grid_height)
        init_true_state = MazeState(start_state[1], start_state[0],
                                    height=grid_height, width=grid_width, coins=coins)

        reward_dict = self.get_rewards()
        # --- Create POMDP problem ---
        problem = MazeProblem(
            planner_name=planner_name,
            goal_state=goal_state,
            solver_config=self.solver_config,
            rewards = reward_dict,
            walls=walls,
            holes=holes,
            traps=traps,
            coins=coins,
            grid_width=grid_width,
            grid_height=grid_height,
            init_belief=init_belief_state,
            init_true_state=init_true_state,
            move_probabilities=self.get_move_probabilites(),
            observation_noises=self.get_observation_noise()
        )

        # --- Simulation setup ---
        step_count = 0
        taken_actions = []
        max_steps = grid_width * grid_height * 2  # prevent infinite loops

        current_state = problem.env.state
        self.display_board()
        self.root.update()

        print("🟢 Initial board:")
        self.print_console_grid(problem.env.state, goal_state, walls, traps, coins)
        heatmap_window = TkinterHeatmapWindow(
            grid_width=problem.width,
            grid_height=problem.height,
            walls=problem.walls,
            traps=problem.traps,
            coins=problem.coins,  # or empty set() if none
            goal_position=problem.goal
        )
        if self.show_heatmap_var.get():
            heatmap_window.update(step_count, problem.get_current_belief_state(),
                                  (problem.env.state.x, problem.env.state.y))

        # --- Main loop ---
        while (current_state.x, current_state.y) != goal_state and step_count < max_steps:
            step_count += 1
            action: Action = problem.take_action()
            taken_actions.append(action.name)
            next_state = MazeState.get_next_state(current_state, action)
            if (next_state.x, next_state.y) in problem.env.walls:
                next_state = current_state
            reward = problem.reward_model.sample(problem.env.state, action, next_state)
            rewards.append(reward)

            # --- Move agent icon on GUI ---
            from_xy = (current_state.x, current_state.y)
            to_xy = (next_state.x, next_state.y)
            self.move_icon_xy(from_xy, to_xy)
            self.root.update()
            time.sleep(0.2)

            # --- Apply transition + update belief ---
            problem.env.apply_transition(next_state)
            obs = problem.observation_model.sample(next_state, action)
            problem.update_belief(action, obs)
            current_state = next_state

            print(f"Step {step_count} | Action={action.name} | Reward={reward}")
            self.print_console_grid(problem.env.state, goal_state, walls, traps, coins)


            # Optional: show histogram of belief
            if self.show_heatmap_var.get():
                heatmap_window.update(step_count, problem.get_current_belief_state(),(problem.env.state.x, problem.env.state.y))

        # --- End simulation ---
        if (current_state.x, current_state.y) == goal_state:
            discounted_total = calculate_discounted_reward(rewards,self.gamma)
            total_reward = calculate_total_sum_reward(rewards)
            print(f"🏁 Goal reached in {step_count} steps!")
            print(f"Total reward sum: {total_reward}")
            print(f"Discounted total reward: {discounted_total}")
        else:
            print(f"⚠ Simulation ended (max {max_steps} steps reached).")

        print(f"Number of actions taken: {len(taken_actions)}")
        print("Actions taken:", taken_actions)
        plt.close('all')
        plt.clf()
        plt.cla()

    def get_rewards(self):
        data = self.board_data
        hole_penalty = data['holes']['penalty']
        trap_penalty = data['traps']['penalty']
        step_cost = data['rewards']['step']
        wall_penalty = data['walls']['penalty']
        goal_reward = data['goal']['reward']
        return {
            'hole_penalty':hole_penalty,
            'trap_penalty':trap_penalty,
            'step_cost':step_cost,
            'wall_penalty':wall_penalty,
            'goal_reward':goal_reward,
        }
    def get_observation_noise(self):
        return self.board_data['observation_noise']
    def get_move_probabilites(self):
        return self.board_data['move_probabilities']