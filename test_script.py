from src.agent.belief import *
from src.domain.action import Action
from src.domain.maze_state import MazeState
from src.problem.problem import MazeProblem

from src.visuals.heatmap_utils import show_histogram

grid_height = 6
grid_width = 6
goal_state = (1, 0)
coins = set()
walls = {(0, 1), (1, 1), (2, 1), (3, 3), (4, 3), (5, 3)}
holes = set()
traps = set()
start_state = (5, 5)
x, y = start_state
init_true_state = MazeState(x, y, height=grid_height, width=grid_width, coins=coins)

init_belief_state = initialize_uniform_histogram_belief(grid_width, grid_height, traps)

problem = MazeProblem("pouct", goal_state, walls, holes, traps, coins,
                      grid_width, grid_height, init_belief_state, init_true_state)

finishing_reward = 0
i = 0
taken_actions = []


def print_grid(agent_state, walls, goal_state, width, height):
    for y in range(height):
        row = ""
        for x in range(width):
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


print_grid(problem.env.state, walls, goal_state, grid_width, grid_height)

while (problem.env.cur_state.x, problem.env.cur_state.y) != goal_state:
    action: Action = problem.take_action()
    taken_actions.append(action.name)
    i += 1
    print("==== Step %d ====" % i)
    print("True state:", problem.env.state)
    print("Action:", action)

    next_state = MazeState.get_next_state(problem.env.state, action)
    current_state = problem.env.cur_state
    if (next_state.x, next_state.y) in problem.env.walls:
        next_state = current_state
    problem.env.apply_transition(next_state)

    real_observation = problem.observation_model.sample(next_state, action)

    reward = problem.reward_model.sample(problem.env.state, action, next_state)

    print("Reward:", reward)
    print("Next state:", next_state)
    print(">> Observation:", real_observation)

    print_grid(problem.env.state, walls, goal_state, grid_width, grid_height)

    problem.update_belief(action, real_observation)

    finishing_reward = reward
    show_histogram(i,
                   problem.get_current_belief_state(),
                   problem.width,
                   problem.height,
                   problem.walls,
                   problem.traps,
                   problem.coins,
                   (problem.env.state.x, problem.env.state.y),
                   problem.goal)

print(taken_actions)
