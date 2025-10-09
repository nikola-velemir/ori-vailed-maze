import time

from src.visuals.game.game import Game


def run_pomdp_simulation(self:Game):
    """Runs a POMDP simulation and visualizes each step."""
    from src.agent.belief import initialize_uniform_histogram_belief
    from src.domain.action import Action
    from src.domain.maze_state import MazeState
    from src.problem.problem import MazeProblem
    from src.visuals.heatmap.heatmap_utils import show_histogram

    # Extract board info
    grid_height = self.rows
    grid_width = self.cols
    goal_state = self.board.find_position('g')
    start_state = self.board.find_position('a')

    # Convert board obstacles
    walls = set(self.board.find_all_positions('w'))
    holes = set(self.board.find_all_positions('h'))
    traps = set(self.board.find_all_positions('t'))
    coins = set()

    # Initialize belief + state
    init_belief_state = initialize_uniform_histogram_belief(grid_width, grid_height, traps)
    init_true_state = MazeState(start_state[1], start_state[0], height=grid_height, width=grid_width, coins=coins)

    # Create POMDP problem
    problem = MazeProblem("pouct", goal_state, walls, holes, traps, coins,
                          grid_width, grid_height, init_belief_state, init_true_state)

    # Simulation parameters
    finishing_reward = 0
    i = 0
    taken_actions = []

    # Visualization helper
    def print_grid(agent_state):
        for y in range(grid_height):
            row = ""
            for x in range(grid_width):
                if agent_state.x == x and agent_state.y == y:
                    row += "A "
                elif goal_state[0] == x and goal_state[1] == y:
                    row += "G "
                elif (x, y) in walls:
                    row += "# "
                elif (x, y) in traps:
                    row += 'T '
                elif (x, y) in coins:
                    row += 'C '
                else:
                    row += ". "
            print(row)
        print()

    print_grid(problem.env.state, )

    # MAIN LOOP
    while (problem.env.cur_state.x, problem.env.cur_state.y) != goal_state:
        action: Action = problem.take_action()
        taken_actions.append(action.name)
        i += 1

        next_state = MazeState.get_next_state(problem.env.state, action)
        problem.env.apply_transition(next_state)

        real_observation = problem.observation_model.sample(next_state, action)
        reward = problem.reward_model.sample(problem.env.state, action, next_state)

        # Update visualization
        old_pos = (problem.env.state.y, problem.env.state.x)
        new_pos = (next_state.y, next_state.x)
        self.move_icon_xy(old_pos, new_pos)
        self.root.update()
        time.sleep(0.5)

        print(f"Step {i} | Action={action.name} | Reward={reward}")
        print_grid(problem.env.state)
        show_histogram(i,
                       problem.get_current_belief_state(),
                       problem.width,
                       problem.height,
                       problem.walls,
                       problem.traps,
                       problem.coins,
                       (problem.env.state.x, problem.env.state.y),
                       problem.goal)

        problem.update_belief(action, real_observation)
        finishing_reward = reward

    print("✅ Reached goal! Actions:", taken_actions)