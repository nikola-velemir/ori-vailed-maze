import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data(csv_file):
    """Load and parse the CSV data."""
    df = pd.read_csv(csv_file)

    # Parse the actions list from string to actual list
    df['actions'] = df['actions'].apply(eval)
    df['path_length'] = df['actions'].apply(len)

    return df


def plot_success_rate_analysis(df, output_dir='plots'):
    """Create success rate analysis plots."""
    Path(output_dir).mkdir(exist_ok=True)

    # 1. Success rate vs maze size
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Success rate by maze size
    success_by_size = df.groupby('maze_size')['reached_goal'].agg(['mean', 'count'])
    success_by_size['success_rate'] = success_by_size['mean'] * 100

    axes[0].bar(success_by_size.index, success_by_size['success_rate'],
                color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Maze Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Success Rate vs. Maze Size', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 105])
    axes[0].grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (idx, row) in enumerate(success_by_size.iterrows()):
        axes[0].text(i, row['success_rate'] + 2, f"n={int(row['count'])}",
                     ha='center', fontsize=10)

    # 2. Overall success distribution (pie chart)
    success_counts = df['reached_goal'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Successful', 'Failed']

    axes[1].pie(success_counts.values, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Overall Success Distribution', fontsize=14, fontweight='bold')

    # 3. Bar chart - successful vs failed
    success_data = df['reached_goal'].value_counts()
    axes[2].bar(['Failed', 'Successful'],
                [success_data.get(False, 0), success_data.get(True, 0)],
                color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Number of Instances', fontsize=12, fontweight='bold')
    axes[2].set_title('Successful vs. Failed Navigation Attempts', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, count in enumerate([success_data.get(False, 0), success_data.get(True, 0)]):
        axes[2].text(i, count + 0.5, str(count), ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/success_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Success rate analysis saved to {output_dir}/success_rate_analysis.png")


def plot_reward_analysis(df, output_dir='plots'):
    """Create reward analysis plots."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Expected reward distribution (histogram)
    axes[0, 0].hist(df['discounted_reward'], bins=30, color='steelblue',
                    alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df['discounted_reward'].mean(), color='red',
                       linestyle='--', linewidth=2, label=f'Mean: {df["discounted_reward"].mean():.2f}')
    axes[0, 0].axvline(df['discounted_reward'].median(), color='green',
                       linestyle='--', linewidth=2, label=f'Median: {df["discounted_reward"].median():.2f}')
    axes[0, 0].set_xlabel('Expected Cumulative Discounted Reward', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Distribution of Expected Rewards', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Reward vs maze size
    maze_sizes = sorted(df['maze_size'].unique())
    reward_by_size = [df[df['maze_size'] == size]['discounted_reward'].values
                      for size in maze_sizes]

    bp = axes[0, 1].boxplot(reward_by_size, labels=maze_sizes, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
    axes[0, 1].set_xlabel('Maze Size', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Expected Reward', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Expected Reward vs. Maze Size', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. Box plot: rewards by success/failure
    success_rewards = df[df['reached_goal'] == True]['discounted_reward']
    failed_rewards = df[df['reached_goal'] == False]['discounted_reward']

    bp_data = [success_rewards, failed_rewards]
    bp = axes[1, 0].boxplot(bp_data, labels=['Successful', 'Failed'],
                            patch_artist=True,
                            boxprops=dict(alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')

    axes[1, 0].set_ylabel('Expected Reward', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Reward Comparison: Successful vs. Failed', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Time series of rewards (by index as proxy for episodes)
    axes[1, 1].plot(df.index, df['discounted_reward'], marker='o',
                    linestyle='-', alpha=0.6, markersize=4)
    axes[1, 1].axhline(df['discounted_reward'].mean(), color='red',
                       linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].set_xlabel('Episode Index', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Expected Reward', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Expected Reward Across Episodes', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/reward_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Reward analysis saved to {output_dir}/reward_analysis.png")


def plot_path_efficiency(df, output_dir='plots'):
    """Create path efficiency metrics plots."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Path length vs maze size
    maze_sizes = sorted(df['maze_size'].unique())
    for size in maze_sizes:
        size_data = df[df['maze_size'] == size]
        axes[0, 0].scatter(size_data['maze_size'], size_data['path_length'],
                           alpha=0.6, s=50, label=f'{size}x{size}')

    # Add trend line
    z = np.polyfit(df['maze_size'], df['path_length'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(df['maze_size'].min(), df['maze_size'].max(), 100)
    axes[0, 0].plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')

    axes[0, 0].set_xlabel('Maze Size', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Path Length (Steps)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Path Length vs. Maze Size', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Histogram of path lengths for successful navigations
    successful_df = df[df['reached_goal'] == True]
    axes[0, 1].hist(successful_df['path_length'], bins=20, color='#2ecc71',
                    alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(successful_df['path_length'].mean(), color='red',
                       linestyle='--', linewidth=2, label=f'Mean: {successful_df["path_length"].mean():.1f}')
    axes[0, 1].set_xlabel('Path Length (Steps)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Path Length Distribution (Successful)', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Scatter: path length vs expected reward
    colors = ['#2ecc71' if reached_goal else '#e74c3c' for reached_goal in df['reached_goal']]
    axes[1, 0].scatter(df['path_length'], df['discounted_reward'],
                       c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Add correlation line
    z = np.polyfit(df['path_length'], df['discounted_reward'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['path_length'].min(), df['path_length'].max(), 100)
    axes[1, 0].plot(x_line, p(x_line), "b--", linewidth=2, alpha=0.8, label='Trend')

    axes[1, 0].set_xlabel('Path Length (Steps)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Expected Reward', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Path Length vs. Expected Reward', fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Successful'),
                       Patch(facecolor='#e74c3c', label='Failed')]
    axes[1, 0].legend(handles=legend_elements)
    axes[1, 0].grid(alpha=0.3)

    # 4. Optimal vs actual path length comparison
    # Calculate theoretical minimum (Manhattan distance would be optimal_steps)
    # For now, use minimum steps per maze size as "optimal"
    optimal_steps = df.groupby('maze_size')['path_length'].min()
    avg_steps = df.groupby('maze_size')['path_length'].mean()

    x = np.arange(len(optimal_steps))
    width = 0.35

    axes[1, 1].bar(x - width / 2, optimal_steps.values, width,
                   label='Best Path', color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 1].bar(x + width / 2, avg_steps.values, width,
                   label='Average Path', color='steelblue', alpha=0.7, edgecolor='black')

    axes[1, 1].set_xlabel('Maze Size', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Path Length (Steps)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Best vs. Average Path Length', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(optimal_steps.index)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/path_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Path efficiency analysis saved to {output_dir}/path_efficiency.png")


def plot_complexity_analysis(df, output_dir='plots'):
    """Create complexity analysis plots."""
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Maze size vs computation time
    time_by_size = df.groupby('maze_size')['time_ms'].agg(['mean', 'std'])

    axes[0].bar(time_by_size.index, time_by_size['mean'],
                yerr=time_by_size['std'], capsize=5,
                color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Maze Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Computation Time (ms)', fontsize=12, fontweight='bold')
    axes[0].set_title('Maze Size vs. Computation Time', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # 2. Maze size vs path length (scalability)
    path_by_size = df.groupby('maze_size')['path_length'].agg(['mean', 'std'])

    axes[1].errorbar(path_by_size.index, path_by_size['mean'],
                     yerr=path_by_size['std'], marker='o', markersize=8,
                     capsize=5, linewidth=2, color='steelblue',
                     markerfacecolor='lightblue', markeredgecolor='black')
    axes[1].set_xlabel('Maze Size', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Path Length (Steps)', fontsize=12, fontweight='bold')
    axes[1].set_title('Scalability: Path Length vs. Maze Size', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    # 3. Success rate decline as complexity increases
    success_rate_by_size = df.groupby('maze_size')['reached_goal'].mean() * 100

    axes[2].plot(success_rate_by_size.index, success_rate_by_size.values,
                 marker='o', markersize=10, linewidth=2, color='steelblue',
                 markerfacecolor='lightblue', markeredgecolor='black')
    axes[2].fill_between(success_rate_by_size.index, success_rate_by_size.values,
                         alpha=0.3, color='steelblue')
    axes[2].set_xlabel('Maze Size', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    axes[2].set_title('Success Rate vs. Maze Complexity', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 105])
    axes[2].grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(success_rate_by_size.index, success_rate_by_size.values, 1)
    p = np.poly1d(z)
    axes[2].plot(success_rate_by_size.index, p(success_rate_by_size.index),
                 "r--", linewidth=2, alpha=0.7, label='Trend')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Complexity analysis saved to {output_dir}/complexity_analysis.png")


def generate_summary_statistics(df, output_dir='plots'):
    """Generate and save summary statistics."""
    Path(output_dir).mkdir(exist_ok=True)

    summary = []
    summary.append("=" * 60)
    summary.append("POMDP MAZE NAVIGATION - SUMMARY STATISTICS")
    summary.append("=" * 60)
    summary.append(f"\nTotal Instances: {len(df)}")
    summary.append(f"Successful: {df['reached_goal'].sum()} ({df['reached_goal'].mean() * 100:.1f}%)")
    summary.append(f"Failed: {(~df['reached_goal']).sum()} ({(~df['reached_goal']).mean() * 100:.1f}%)")

    summary.append("\n" + "-" * 60)
    summary.append("REWARD STATISTICS")
    summary.append("-" * 60)
    summary.append(f"Mean Expected Reward: {df['discounted_reward'].mean():.2f}")
    summary.append(f"Median Expected Reward: {df['discounted_reward'].median():.2f}")
    summary.append(f"Std Dev: {df['discounted_reward'].std():.2f}")
    summary.append(f"Min: {df['discounted_reward'].min():.2f}, Max: {df['discounted_reward'].max():.2f}")

    summary.append("\n" + "-" * 60)
    summary.append("PATH LENGTH STATISTICS")
    summary.append("-" * 60)
    summary.append(f"Mean Path Length: {df['path_length'].mean():.1f}")
    summary.append(f"Median Path Length: {df['path_length'].median():.1f}")
    summary.append(f"Std Dev: {df['path_length'].std():.1f}")
    summary.append(f"Min: {df['path_length'].min()}, Max: {df['path_length'].max()}")

    summary.append("\n" + "-" * 60)
    summary.append("COMPUTATION TIME STATISTICS")
    summary.append("-" * 60)
    summary.append(f"Mean Time: {df['time_ms'].mean():.2f} ms")
    summary.append(f"Median Time: {df['time_ms'].median():.2f} ms")
    summary.append(f"Std Dev: {df['time_ms'].std():.2f} ms")
    summary.append(f"Min: {df['time_ms'].min():.2f} ms, Max: {df['time_ms'].max():.2f} ms")

    summary.append("\n" + "-" * 60)
    summary.append("PERFORMANCE BY MAZE SIZE")
    summary.append("-" * 60)
    for size in sorted(df['maze_size'].unique()):
        size_df = df[df['maze_size'] == size]
        success_rate = size_df['reached_goal'].mean() * 100
        avg_reward = size_df['discounted_reward'].mean()
        avg_steps = size_df['path_length'].mean()
        avg_time = size_df['time_ms'].mean()

        summary.append(f"\nMaze Size {size}x{size}:")
        summary.append(f"  Instances: {len(size_df)}")
        summary.append(f"  Success Rate: {success_rate:.1f}%")
        summary.append(f"  Avg Reward: {avg_reward:.2f}")
        summary.append(f"  Avg Steps: {avg_steps:.1f}")
        summary.append(f"  Avg Time: {avg_time:.2f} ms")

    summary.append("\n" + "=" * 60)

    summary_text = "\n".join(summary)

    # Print to console
    print(summary_text)

    # Save to file
    with open(f'{output_dir}/summary_statistics.txt', 'w') as f:
        f.write(summary_text)

    print(f"\n✓ Summary statistics saved to {output_dir}/summary_statistics.txt")


def main():
    """Main function to generate all plots."""
    import sys

    # Load data
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'results/test/test_data_1760370450.8296359.csv'  # Default file name

    print(f"Loading data from {csv_file}...")
    try:
        df = load_data(csv_file)
        print(f"✓ Loaded {len(df)} instances\n")
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found!")
        print("Usage: python script.py [csv_file_path]")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create output directory
    output_dir = 'plots'
    Path(output_dir).mkdir(exist_ok=True)

    # Generate all plots
    print("Generating plots...\n")
    plot_success_rate_analysis(df, output_dir)
    plot_reward_analysis(df, output_dir)
    plot_path_efficiency(df, output_dir)
    plot_complexity_analysis(df, output_dir)

    # Generate summary statistics
    print("\nGenerating summary statistics...\n")
    generate_summary_statistics(df, output_dir)

    print(f"\n{'=' * 60}")
    print("All plots generated successfully!")
    print(f"Check the '{output_dir}' directory for results.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()