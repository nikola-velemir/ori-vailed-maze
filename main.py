import pomdp_py

from src.agent.belief import initialize_default_belief_state
from src.domain.action import Action
from src.domain.maze_state import MazeState
from src.problem import MazeProblem

grid_height = 6
grid_width = 6
goal_state = (0, 0)
init_true_state = MazeState(5, 5)
init_belief_state = initialize_default_belief_state(grid_width, grid_height)

walls = {(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)}
problem = MazeProblem(goal_state, walls, grid_width, grid_height, init_belief_state, init_true_state)

planner = pomdp_py.POUCT(max_depth=10, discount_factor=0.95,
                         exploration_const=110, planning_time=2,
                         rollout_policy=problem.agent.policy_model)
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
                row += "# "  # wall
            else:
                row += ". "  # empty space
        print(row)
    print()


print_grid(problem.env.state, walls, goal_state, grid_width, grid_height)

while finishing_reward != 100:
    action: Action = planner.plan(problem.agent)
    taken_actions.append(action.name)
    i += 1
    print("==== Step %d ====" % i)
    print("True state:", problem.env.state)
    print("Action:", action)

    next_state = MazeState.get_next_state(problem.env.state, action)
    problem.env.apply_transition(next_state)

    real_observation = problem.observation_model.sample(next_state, action)

    reward = problem.reward_model.sample(problem.env.state, action, next_state)

    print("Reward:", reward)
    print("Next state:", next_state)
    print(">> Observation:", real_observation)

    print_grid(problem.env.state, walls, goal_state, grid_width, grid_height)

    problem.agent.update_history(action, real_observation)
    planner.update(problem.agent, action, real_observation)

    if isinstance(planner, pomdp_py.POUCT):
        print("Num sims:", planner.last_num_sims)

    if isinstance(problem.agent.cur_belief, pomdp_py.Histogram):
        new_belief = pomdp_py.update_histogram_belief(
            problem.agent.cur_belief,
            action, real_observation,
            problem.agent.observation_model,
            problem.agent.transition_model
        )
        problem.agent.set_belief(new_belief)

    finishing_reward = reward

print(taken_actions)
