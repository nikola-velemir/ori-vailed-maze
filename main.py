import pomdp_py
import numpy as np
from src.agent.belief import *
from src.domain.action import Action
from src.domain.maze_state import MazeState
from src.problem import MazeProblem
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

init_belief_state = initialize_default_belief_state(grid_width, grid_height, traps)

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


def show_histogram(step):
    belief: pomdp_py.Histogram = problem.get_current_belief_state()
    heatmap = np.zeros((grid_height, grid_width))

    # Correct iteration over Histogram
    for state in belief:
        prob = belief[state]
        heatmap[state.y, state.x] += prob

    # Create fresh figure each time
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a better colormap and normalization
    im = ax.imshow(
        heatmap,
        origin='upper',
        cmap='YlOrRd',
        interpolation='nearest',
        vmin=0,
        vmax=max(heatmap.max(), 0.01),
        aspect='equal'
    )

    # Add gridlines for better cell visibility
    ax.set_xticks(np.arange(grid_width + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_height + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.tick_params(which='minor', size=0)

    # Major ticks for labels
    ax.set_xticks(range(grid_width))
    ax.set_yticks(range(grid_height))

    # Draw walls as gray rectangles
    for (wx, wy) in walls:
        rect = Rectangle((wx - 0.5, wy - 0.5), 1, 1,
                         facecolor='gray', edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)

    # Draw traps if any
    for (tx, ty) in traps:
        rect = Rectangle((tx - 0.5, ty - 0.5), 1, 1,
                         facecolor='purple', edgecolor='black', linewidth=2, alpha=0.6)
        ax.add_patch(rect)

    # Draw coins if any
    for (cx, cy) in coins:
        ax.scatter(cx, cy, color='gold', s=150, marker='o',
                   edgecolors='orange', linewidths=2, label='Coin', zorder=5)

    # Overlay goal with star
    ax.scatter(goal_state[0], goal_state[1],
               color='lime', marker='*', s=400,
               edgecolors='darkgreen', linewidths=2, label='Goal', zorder=6)

    # Overlay agent position
    ax.scatter(problem.env.state.x, problem.env.state.y,
               color='blue', s=200, marker='o',
               edgecolors='darkblue', linewidths=2, label='Agent', zorder=7)

    # Add probability text in cells (only if significant)
    for y in range(grid_height):
        for x in range(grid_width):
            if heatmap[y, x] > 0.01:
                ax.text(x, y, f'{heatmap[y, x]:.2f}',
                        ha='center', va='center',
                        color='white' if heatmap[y, x] > 0.5 else 'black',
                        fontsize=8, fontweight='bold')

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Belief Probability')

    ax.set_title(f'POMCP Belief Distribution at Step {step}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)

    # Update legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(),
                  loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


while finishing_reward != 100:
    action: Action = problem.take_action()
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

    problem.update_belief(action, real_observation)

    finishing_reward = reward
    show_histogram(i)

print(taken_actions)

plt.ioff()
plt.show()
