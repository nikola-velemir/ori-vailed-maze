import pandas as pd


def display_summary(df: pd.DataFrame):
    print("\n✅ Batch run summary:")
    print(f"- Total episodes: {len(df)}")
    if 'total_reward' in df.columns:
        print(f"- Average total reward: {df['total_reward'].mean():.2f}")
    if 'discounted_reward' in df.columns:
        print(f"- Average discounted reward: {df['discounted_reward'].mean():.2f}")
    if 'steps' in df.columns:
        print(f"- Average steps: {df['steps'].mean():.2f}")
    if 'path_length' in df.columns:
        print(f"- Average path length: {df['path_length'].mean():.2f}")
    if 'reached_goal' in df.columns:
        print(f"- Success rate: {df['reached_goal'].mean() * 100:.1f}%")

    print("\nFirst few results:")
    print(df.head())
