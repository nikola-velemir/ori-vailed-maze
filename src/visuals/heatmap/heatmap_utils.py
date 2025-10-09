import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def show_histogram(step, belief,grid_width, grid_height,walls, traps, coins,current_position, goal_position):
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
    ax.scatter(goal_position[0], goal_position[1],
               color='lime', marker='*', s=400,
               edgecolors='darkgreen', linewidths=2, label='Goal', zorder=6)

    # Overlay agent position
    ax.scatter(current_position[0],current_position[1],
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