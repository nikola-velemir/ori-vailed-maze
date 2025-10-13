from src.agent.belief import initialize_uniform_belief
from src.board_parser.board_parser import BoardParser
from src.utils.metric_utils import calculate_discounted_reward, calculate_total_sum_reward, calculate_path_length

from src.domain.action import Action
from src.domain.maze_state import MazeState
from src.problem.problem import MazeProblem


def run_training(test_file: str, headless: bool = False):
    data = BoardParser.parse(test_file)
    rewards = []

    grid_width = data['width']
    grid_height = data['height']
    r, c = data['goal']['position']
    goal_state = (r, c)
    c, r = data['agent']
    start_state = (c, r)

    if not goal_state or not start_state:
        print("❌ Missing agent or goal position on the board.")
        return

    walls = set((c, r) for r, c in data.get("walls", {}).get("positions", []))
    holes = set(tuple(pos) for pos in data.get("holes", {}).get("positions", []))
    traps = set(tuple(pos) for pos in data.get("traps", {}).get("positions", []))
    coins = set(tuple(pos) for pos in data.get("coins", {}).get("positions", []))

    # --- Initialize belief + state ---
    init_belief_state = initialize_uniform_belief('pouct', grid_width, grid_height, coins)
    init_true_state = MazeState(start_state[1], start_state[0],
                                height=grid_height, width=grid_width, coins=coins)

    solver_config = data['solver_config']
    reward_dict = get_rewards(data)
    # --- Create POMDP problem ---
    problem = MazeProblem(
        planner_name='pouct',
        goal_state=goal_state,
        solver_config=solver_config,
        rewards=reward_dict,
        walls=walls,
        holes=holes,
        traps=traps,
        coins=coins,
        grid_width=grid_width,
        grid_height=grid_height,
        init_belief=init_belief_state,
        init_true_state=init_true_state,
        move_probabilities=get_move_probabilites(data),
        observation_noises=get_observation_noise(data)
    )

    # --- Simulation setup ---
    step_count = 0
    taken_actions = []
    max_steps = grid_width * grid_height * 2  # prevent infinite loops

    current_state = problem.env.state

    print("🟢 Initial board:")
    reached_goal = False
    print_console_grid(grid_width, grid_height, problem.env.state, goal_state, walls, traps, coins, headless=headless)

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

        # --- Apply transition + update belief ---
        problem.env.apply_transition(next_state)
        obs = problem.observation_model.sample(next_state, action)
        problem.update_belief(action, obs)
        current_state = next_state

        print(f"Step {step_count} | Action={action.name} | Reward={reward}")
        print_console_grid(grid_width, grid_height, problem.env.state, goal_state, walls, traps, coins,
                           headless=headless)

    path_length = calculate_path_length(taken_actions)
    # --- End simulation ---
    if (current_state.x, current_state.y) == goal_state:
        discounted_total = calculate_discounted_reward(rewards, solver_config['discount_factor'])
        total_reward = calculate_total_sum_reward(rewards)
        print(f"🏁 Goal reached in {step_count} steps!")
        print(f"Total reward sum: {total_reward}")
        print(f"Discounted total reward: {discounted_total}")
        reached_goal = True
    else:
        print(f"⚠ Simulation ended (max {max_steps} steps reached).")

    print(f"Number of actions taken: {path_length}")
    print("Actions taken:", taken_actions)
    return {
        'file': test_file,
        "total_reward": calculate_total_sum_reward(rewards),
        "discounted_reward": calculate_discounted_reward(rewards),
        "path_length": path_length,
        "actions": taken_actions,
        "reached_goal": reached_goal,
        'maze_size': f'{data['width']}x{data['height']}',
    }


def get_rewards(data):
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


def get_observation_noise(data):
    return data['observation_noise']


def get_move_probabilites(data):
    return data['move_probabilities']


def print_console_grid(grid_width, grid_height, agent_state, goal_state, walls, traps, coins, headless=False):
    if headless:
        return
    print()
    for y in range(grid_height):
        row = ""
        for x in range(grid_width):
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


if __name__ == '__main__':
    run_training("dataset/train/board_0b8087fe-92fd-4e4e-bb61-8badd011a238.json")
