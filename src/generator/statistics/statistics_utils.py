import os.path
import uuid

import matplotlib.pyplot as plt
import seaborn as sns
import ast
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def run_basic_assessment(df):
    print("=" * 60)
    print("PERFORMANCE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nTotal test cases: {len(df)}")
    print(f"Successfully reached goal: {df['reached_goal'].sum()} ({df['reached_goal'].mean() * 100:.1f}%)")
    print(f"Failed to reach goal: {(~df['reached_goal']).sum()} ({(~df['reached_goal']).mean() * 100:.1f}%)")
    print(f"\nAverage path length: {df['path_length'].mean():.2f}")
    print(f"Average total reward: {df['total_reward'].mean():.2f}")
    print(f"Average discounted reward: {df['discounted_reward'].mean():.2f}")
    print("=" * 60)


def sumerize_success(df):
    # Separate successful and failed cases
    successful = df[df['reached_goal'] == True]
    failed = df[df['reached_goal'] == False]

    print(f"\n--- Successful Cases ---")
    print(f"Average path length: {successful['path_length'].mean():.2f}")
    print(f"Average total reward: {successful['total_reward'].mean():.2f}")
    print(f"Average discounted reward: {successful['discounted_reward'].mean():.2f}")

    print(f"\n--- Failed Cases ---")
    print(f"Average path length: {failed['path_length'].mean():.2f}")
    print(f"Average total reward: {failed['total_reward'].mean():.2f}")
    print(f"Average discounted reward: {failed['discounted_reward'].mean():.2f}")

    ax1 = plt.subplot(4, 3, 1)
    success_counts = df['reached_goal'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie(success_counts, labels=['Success', 'Failed'], autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Success Rate', fontsize=14, fontweight='bold')
    return successful, failed


def summarize_path_length_distribution(df, successful, failed):
    # 2. Path Length Distribution
    ax2 = plt.subplot(4, 3, 2)
    ax2.hist(successful['path_length'], bins=15, alpha=0.7, label='Success', color='#2ecc71', edgecolor='black')
    ax2.hist(failed['path_length'], bins=15, alpha=0.7, label='Failed', color='#e74c3c', edgecolor='black')
    ax2.set_xlabel('Path Length')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Path Length Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)


def summarize_total_reward_distribution(successful, failed):
    ax3 = plt.subplot(4, 3, 3)
    ax3.hist(successful['total_reward'], bins=15, alpha=0.7, label='Success', color='#2ecc71', edgecolor='black')
    ax3.hist(failed['total_reward'], bins=15, alpha=0.7, label='Failed', color='#e74c3c', edgecolor='black')
    ax3.set_xlabel('Total Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Total Reward Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)


def summarize_discounted_reward_distribution(successful, failed):
    ax5 = plt.subplot(4, 3, 5)
    ax5.hist(successful['discounted_reward'], bins=15, alpha=0.7, label='Success', color='#2ecc71', edgecolor='black')
    ax5.hist(failed['discounted_reward'], bins=15, alpha=0.7, label='Failed', color='#e74c3c', edgecolor='black')
    ax5.set_xlabel('Discounted Reward')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Discounted Reward Distribution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)


def summarize_path_length_to_total_reward(successful, failed):
    ax4 = plt.subplot(4, 3, 4)
    ax4.scatter(successful['path_length'], successful['total_reward'],
                alpha=0.6, s=100, c='#2ecc71', label='Success', edgecolors='black', linewidth=0.5)
    ax4.scatter(failed['path_length'], failed['total_reward'],
                alpha=0.6, s=100, c='#e74c3c', label='Failed', edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Path Length')
    ax4.set_ylabel('Total Reward')
    ax4.set_title('Path Length vs Total Reward', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)


def summarize_performance_over_test_cases(df, colors_scatter):
    ax6 = plt.subplot(4, 3, 6)
    ax6.scatter(range(len(df)), df['total_reward'], c=colors_scatter, alpha=0.6, s=100, edgecolors='black',
                linewidth=0.5)
    ax6.set_xlabel('Test Case Index')
    ax6.set_ylabel('Total Reward')
    ax6.set_title('Total Reward Across Test Cases', fontsize=14, fontweight='bold')
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax6.grid(True, alpha=0.3)


def summarize_path_length_over_test_cases(df, colors_scatter):
    ax7 = plt.subplot(4, 3, 7)
    ax7.scatter(range(len(df)), df['path_length'], c=colors_scatter, alpha=0.6, s=100, edgecolors='black',
                linewidth=0.5)
    ax7.set_xlabel('Test Case Index')
    ax7.set_ylabel('Path Length')
    ax7.set_title('Path Length Across Test Cases', fontsize=14, fontweight='bold')
    ax7.axhline(y=128, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Max Length (128)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)


def summarize_reward_efficiency(df):
    ax9 = plt.subplot(4, 3, 9)
    df['reward_per_step'] = df['total_reward'] / df['path_length'].replace(0, 1)
    successful_eff = df[df['reached_goal'] == True]['reward_per_step']
    failed_eff = df[df['reached_goal'] == False]['reward_per_step']
    ax9.hist(successful_eff, bins=15, alpha=0.7, label='Success', color='#2ecc71', edgecolor='black')
    ax9.hist(failed_eff, bins=15, alpha=0.7, label='Failed', color='#e74c3c', edgecolor='black')
    ax9.set_xlabel('Reward per Step')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Navigation Efficiency', fontsize=14, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)


def make_path_length_box_plot(successful, failed):
    ax8 = plt.subplot(4, 3, 8)
    box_data = [successful['path_length'], failed['path_length']]
    bp = ax8.boxplot(box_data, labels=['Success', 'Failed'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    ax8.set_ylabel('Path Length')
    ax8.set_title('Path Length Comparison', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')


# NEW FUNCTIONS - Missing visualizations
def analyze_action_distribution(df):
    df['actions_list'] = df['actions'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else [])

    action_counts = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
    for actions in df['actions_list']:
        for action in actions:
            if action in action_counts:
                action_counts[action] += 1

    print(f"\n--- Action Distribution ---")
    total_actions = sum(action_counts.values())
    for action, count in action_counts.items():
        print(f"{action}: {count} ({count / total_actions * 100:.1f}%)")

    ax10 = plt.subplot(4, 3, 10)
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    colors_bar = ['#3498db', '#9b59b6', '#f39c12', '#e67e22']
    bars = ax10.bar(actions, counts, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax10.set_xlabel('Action')
    ax10.set_ylabel('Count')
    ax10.set_title('Action Distribution', fontsize=14, fontweight='bold')
    ax10.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width() / 2., height,
                  f'{int(height)}',
                  ha='center', va='bottom', fontweight='bold')

    return action_counts


def visualize_discounted_vs_total_reward(df, colors_scatter):
    ax11 = plt.subplot(4, 3, 11)
    ax11.scatter(df['total_reward'], df['discounted_reward'], c=colors_scatter,
                 alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    ax11.set_xlabel('Total Reward')
    ax11.set_ylabel('Discounted Reward')
    ax11.set_title('Total vs Discounted Reward', fontsize=14, fontweight='bold')
    ax11.grid(True, alpha=0.3)


def visualize_success_rate_by_path_range(df):
    ax12 = plt.subplot(4, 3, 12)
    path_bins = [0, 10, 20, 50, 128]
    path_labels = ['0-10', '11-20', '21-50', '51-128']
    df['path_range'] = pd.cut(df['path_length'], bins=path_bins, labels=path_labels)
    success_by_range = df.groupby('path_range')['reached_goal'].agg(['sum', 'count'])
    success_rate_by_range = (success_by_range['sum'] / success_by_range['count'] * 100).fillna(0)

    bars = ax12.bar(range(len(success_rate_by_range)), success_rate_by_range,
                    color='#3498db', edgecolor='black', linewidth=1.5)
    ax12.set_xticks(range(len(path_labels)))
    ax12.set_xticklabels(path_labels)
    ax12.set_xlabel('Path Length Range')
    ax12.set_ylabel('Success Rate (%)')
    ax12.set_title('Success Rate by Path Length', fontsize=14, fontweight='bold')
    ax12.grid(True, alpha=0.3, axis='y')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax12.text(bar.get_x() + bar.get_width() / 2., height,
                  f'{height:.1f}%',
                  ha='center', va='bottom', fontweight='bold')


def run_detailed_assessment(df, successful, failed):
    print("\n" + "=" * 60)
    print("DETAILED STATISTICS")
    print("=" * 60)

    print("\n--- Path Length Statistics ---")
    print(f"Min: {df['path_length'].min()}")
    print(f"Max: {df['path_length'].max()}")
    print(f"Median: {df['path_length'].median()}")
    print(f"Std Dev: {df['path_length'].std():.2f}")

    print("\n--- Total Reward Statistics ---")
    print(f"Min: {df['total_reward'].min()}")
    print(f"Max: {df['total_reward'].max()}")
    print(f"Median: {df['total_reward'].median():.2f}")
    print(f"Std Dev: {df['total_reward'].std():.2f}")

    print("\n--- Discounted Reward Statistics ---")
    print(f"Min: {df['discounted_reward'].min():.2f}")
    print(f"Max: {df['discounted_reward'].max():.2f}")
    print(f"Median: {df['discounted_reward'].median():.2f}")
    print(f"Std Dev: {df['discounted_reward'].std():.2f}")

    print("\n" + "=" * 60)
    print("BEST AND WORST CASES")
    print("=" * 60)

    print("\n--- Top 5 Best Performing (Highest Total Reward) ---")
    top_5 = df.nlargest(5, 'total_reward')[['file', 'total_reward', 'path_length', 'reached_goal']]
    for idx, row in top_5.iterrows():
        print(f"{row['file']}: Reward={row['total_reward']}, Path={row['path_length']}, Success={row['reached_goal']}")

    print("\n--- Top 5 Worst Performing (Lowest Total Reward) ---")
    bottom_5 = df.nsmallest(5, 'total_reward')[['file', 'total_reward', 'path_length', 'reached_goal']]
    for idx, row in bottom_5.iterrows():
        print(f"{row['file']}: Reward={row['total_reward']}, Path={row['path_length']}, Success={row['reached_goal']}")

    print("\n--- Most Efficient (Shortest Successful Path) ---")
    if len(successful) > 0:
        efficient = successful.nsmallest(5, 'path_length')[['file', 'total_reward', 'path_length']]
        for idx, row in efficient.iterrows():
            print(f"{row['file']}: Path={row['path_length']}, Reward={row['total_reward']}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


def run_statistics(df):
    # Basic statistics
    run_basic_assessment(df)

    # Create comprehensive visualization (changed to 4x3 layout)
    fig = plt.figure(figsize=(18, 16))

    successful, failed = sumerize_success(df)
    summarize_path_length_distribution(df, successful, failed)

    summarize_total_reward_distribution(successful, failed)

    summarize_path_length_to_total_reward(successful, failed)

    summarize_discounted_reward_distribution(successful, failed)

    colors_scatter = ['#2ecc71' if x else '#e74c3c' for x in df['reached_goal']]

    summarize_performance_over_test_cases(df, colors_scatter)

    summarize_path_length_over_test_cases(df, colors_scatter)

    make_path_length_box_plot(successful, failed)

    summarize_reward_efficiency(df)

    # NEW: Add missing visualizations
    analyze_action_distribution(df)

    visualize_discounted_vs_total_reward(df, colors_scatter)

    visualize_success_rate_by_path_range(df)

    plt.tight_layout()
    file_dir = os.path.join('statistics','results')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_id = uuid.uuid4()
    file_path = os.path.join(file_dir,f'maze_performance_evaluation_{file_id}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as 'maze_performance_evaluation.png'")
    plt.show()

    run_detailed_assessment(df, successful, failed)