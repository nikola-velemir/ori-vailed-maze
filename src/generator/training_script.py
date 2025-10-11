import glob

import pandas as pd

from src.generator.runner import run_training


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

    # Optional: show first few rows
    print("\nFirst few results:")
    print(df.head())


if __name__ == "__main__":
    results = []
    episode = 1
    for file in glob.glob("dataset/train/*.json"):
        res = run_training(file, headless=True)
        results.append(res)
        print(f"{episode}. Episode")
        episode += 1

    df = pd.DataFrame(results)
    df.to_csv("training_data.csv", index=False)

    display_summary(df=df)
