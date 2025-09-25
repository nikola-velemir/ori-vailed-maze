import pomdp_py

from src.agent.belief import inititalize_default_belief_state
from src.domain.maze_state import MazeState
from src.domain.observation import Observation
from src.problem import MazeProblem

grid_height = 5
grid_width = 5

init_true_state = MazeState(0, 0)
init_belief_state = inititalize_default_belief_state(grid_width, grid_height)

problem = MazeProblem((4, 4), grid_width, grid_height, init_belief_state, init_true_state)

pouct = pomdp_py.POUCT(max_depth=10, discount_factor=0.95,
                       planning_time=2, exploration_const=110,
                       rollout_policy=problem.agent.policy_model)
finishing_reward = 0
i=0
while finishing_reward != 100:
    action = pouct.plan(problem.agent)
    print("==== Step %d ====" % (i + 1))
    print("True state:", problem.env.state)
    print("Action:", action)

    # Sample next state using transition model
    next_state = problem.transition_model.sample(problem.env.state, action)
    problem.env.apply_transition(next_state)

    # Sample observation from next state
    real_observation = problem.observation_model.sample(next_state, action)

    # Get reward
    reward = problem.reward_model.sample(problem.env.state, action, next_state)

    # Print results
    print("Reward:", reward)
    print("Next state:", next_state)
    print(">> Observation:", real_observation)



    # Update history and planner
    problem.agent.update_history(action, real_observation)
    pouct.update(problem.agent, action, real_observation)

    if isinstance(pouct, pomdp_py.POUCT):
        print("Num sims:", pouct.last_num_sims)

    # Update belief
    if isinstance(problem.agent.cur_belief, pomdp_py.Histogram):
        new_belief = pomdp_py.update_histogram_belief(
            problem.agent.cur_belief,
            action, real_observation,
            problem.agent.observation_model,
            problem.agent.transition_model
        )
        problem.agent.set_belief(new_belief)

    finishing_reward = reward
