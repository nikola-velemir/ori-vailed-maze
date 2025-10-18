import time

from matplotlib import pyplot as plt

from src.agent.belief import initialize_uniform_belief
from src.utils.metric_utils import calculate_discounted_reward, calculate_total_sum_reward, calculate_path_length
from src.visuals.heatmap.heatmap_window import TkinterHeatmapWindow


def run_pomdp_simulation(self):
    rewards = []
    planner_name = self.search_class_text.get()
    self.reset()
    """Run a POMDP simulation and visually update the board after each step."""
    from src.domain.action import Action
    from src.domain.maze_state import MazeState
    from src.problem.problem import MazeProblem

    print("\n▶ Starting POMDP simulation...")

    # --- Extract board setup ---
    grid_height = self.rows
    grid_width = self.cols
    r, c = self.board.find_position('g')
    goal_state = (c, r)
    c, r = self.board.find_position('a')
    start_state = (c, r)
    if planner_name.lower() =='pomcp':
        planner_name = 'pouct'

    if not goal_state or not start_state:
        print("❌ Missing agent or goal position on the board.")
        return

    walls = set((c, r) for r, c in self.board.find_all_positions('w'))
    holes = set((c, r) for r, c in self.board.find_all_positions('h'))
    traps = set((c, r) for r, c in self.board.find_all_positions('t'))
    coins = set((c, r) for r, c in self.board.find_all_positions('c'))

    # --- Initialize belief + state ---
    init_belief_state = initialize_uniform_belief(planner_name, grid_width, grid_height, coins)
    init_true_state = MazeState(start_state[1], start_state[0],
                                height=grid_height, width=grid_width, coins=coins)

    reward_dict = self.get_rewards()
    # --- Create POMDP problem ---
    problem = MazeProblem(
        planner_name=planner_name,
        goal_state=goal_state,
        solver_config=self.solver_config,
        rewards=reward_dict,
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
    heatmap_window = None
    if self.show_heatmap_var.get():
        if not heatmap_window:
            heatmap_window = TkinterHeatmapWindow(
                grid_width=problem.width,
                grid_height=problem.height,
                walls=problem.walls,
                traps=problem.traps,
                coins=problem.coins,  # or empty set() if none
                goal_position=problem.goal
            )
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
        (old_state_x, old_state_y) = current_state.x, current_state.y
        (new_state_x, new_state_y) = next_state.x, next_state.y

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
            if not heatmap_window:
                heatmap_window = TkinterHeatmapWindow(
                    grid_width=problem.width,
                    grid_height=problem.height,
                    walls=problem.walls,
                    traps=problem.traps,
                    coins=problem.coins,  # or empty set() if none
                    goal_position=problem.goal
                )
            heatmap_window.update(step_count, problem.get_current_belief_state(),
                                  (problem.env.state.x, problem.env.state.y))
        self.add_move_to_history(action.name, (old_state_x,old_state_y),(new_state_x,new_state_y), reward)

    path_length = calculate_path_length(taken_actions)
    # --- End simulation ---
    if (current_state.x, current_state.y) == goal_state:
        discounted_total = calculate_discounted_reward(rewards, self.gamma)
        total_reward = calculate_total_sum_reward(rewards)
        print(f"🏁 Goal reached in {step_count} steps!")
        print(f"Total reward sum: {total_reward}")
        print(f"Discounted total reward: {discounted_total}")
    else:
        print(f"⚠ Simulation ended (max {max_steps} steps reached).")

    print(f"Number of actions taken: {path_length}")
    print("Actions taken:", taken_actions)
    plt.close('all')
    plt.clf()
    plt.cla()
