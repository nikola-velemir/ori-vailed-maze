import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import gc


class TkinterHeatmapWindow:
    """Displays heatmap in a Tkinter window with matplotlib embedded."""

    def __init__(self, grid_width, grid_height, walls, traps, coins, goal_position):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.walls = walls
        self.traps = traps
        self.coins = coins
        self.goal_position = goal_position

        # Create Tkinter window
        self.window = tk.Tk()
        self.window.title("POMCP Belief Heatmap")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)

        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Store objects for updates
        self.im = None
        self.colorbar = None
        self.agent_scatter = None
        self.goal_scatter = None
        self.coin_scatters = []
        self.text_objects = []
        self.wall_patches = []
        self.trap_patches = []

        # Setup static elements
        self._setup_grid()
        self._setup_static_elements()

        self.is_open = True

    def _setup_grid(self):
        """Setup grid lines and ticks."""
        self.ax.set_xticks(np.arange(self.grid_width + 1) - 0.5, minor=True)
        self.ax.set_yticks(np.arange(self.grid_height + 1) - 0.5, minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.5)
        self.ax.tick_params(which='minor', size=0)

        self.ax.set_xticks(range(self.grid_width))
        self.ax.set_yticks(range(self.grid_height))

        self.ax.set_xlabel('X Coordinate', fontsize=12)
        self.ax.set_ylabel('Y Coordinate', fontsize=12)

    def _setup_static_elements(self):
        """Draw walls, traps, coins, and goal."""
        # Draw walls
        for (wx, wy) in self.walls:
            rect = Rectangle((wx - 0.5, wy - 0.5), 1, 1,
                             facecolor='gray', edgecolor='black',
                             linewidth=2, alpha=0.8, zorder=2)
            self.ax.add_patch(rect)
            self.wall_patches.append(rect)

        # Draw traps
        for (tx, ty) in self.traps:
            rect = Rectangle((tx - 0.5, ty - 0.5), 1, 1,
                             facecolor='purple', edgecolor='black',
                             linewidth=2, alpha=0.6, zorder=2)
            self.ax.add_patch(rect)
            self.trap_patches.append(rect)

        # Draw coins
        for (cx, cy) in self.coins:
            scatter = self.ax.scatter(cx, cy, color='gold', s=150, marker='o',
                                      edgecolors='orange', linewidths=2,
                                      label='Coin', zorder=5)
            self.coin_scatters.append(scatter)

        # Draw goal
        self.goal_scatter = self.ax.scatter(
            self.goal_position[0], self.goal_position[1],
            color='lime', marker='*', s=400,
            edgecolors='darkgreen', linewidths=2,
            label='Goal', zorder=6
        )

    def update(self, step, belief, current_position):
        """Update the heatmap with new belief and agent position."""
        if not self.is_open:
            return

        # Compute heatmap
        heatmap = np.zeros((self.grid_height, self.grid_width))
        for state in belief:
            prob = belief[state]
            heatmap[state.y, state.x] += prob

        # Update or create heatmap image
        if self.im is None:
            self.im = self.ax.imshow(
                heatmap,
                origin='upper',
                cmap='YlOrRd',
                interpolation='nearest',
                vmin=0,
                vmax=max(heatmap.max(), 0.01),
                aspect='equal',
                zorder=1
            )
            # Add colorbar once
            self.colorbar = self.fig.colorbar(self.im, ax=self.ax,
                                              fraction=0.046, pad=0.04,
                                              label='Belief Probability')
        else:
            # Just update the data
            self.im.set_data(heatmap)
            self.im.set_clim(vmin=0, vmax=max(heatmap.max(), 0.01))

        # Clear old text objects
        for text in self.text_objects:
            text.remove()
        self.text_objects.clear()

        # Add probability text in cells (only if significant)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if heatmap[y, x] > 0.01:
                    text = self.ax.text(
                        x, y, f'{heatmap[y, x]:.2f}',
                        ha='center', va='center',
                        color='white' if heatmap[y, x] > 0.5 else 'black',
                        fontsize=8, fontweight='bold', zorder=8
                    )
                    self.text_objects.append(text)

        # Update agent position
        if self.agent_scatter is None:
            self.agent_scatter = self.ax.scatter(
                current_position[0], current_position[1],
                color='blue', s=200, marker='o',
                edgecolors='darkblue', linewidths=2,
                label='Agent', zorder=7
            )
        else:
            self.agent_scatter.set_offsets([current_position])

        # Update title
        self.ax.set_title(
            f'POMCP Belief Distribution at Step {step}',
            fontsize=14, fontweight='bold'
        )

        # Update legend (only if not already set)
        if not self.ax.get_legend():
            handles, labels = self.ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                self.ax.legend(by_label.values(), by_label.keys(),
                               loc='upper left', bbox_to_anchor=(1.15, 1),
                               fontsize=10)

        # Redraw canvas
        self.canvas.draw()
        self.window.update()

    def on_closing(self):
        """Handle window close event."""
        self.is_open = False
        self.window.destroy()

    def close(self):
        """Clean up resources."""
        if self.is_open:
            self.window.destroy()
            self.is_open = False
        gc.collect()

