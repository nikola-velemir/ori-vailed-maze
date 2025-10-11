import itertools
import statistics

from src.generator.test_runner import run_all_tests


def evaluate_solver_configs():
    configs = [
        {"num_sims": 100, "exploration_const": 1.0, "discount_factor": 0.95},
        {"num_sims": 300, "exploration_const": 1.5, "discount_factor": 0.99},
        {"num_sims": 500, "exploration_const": 0.7, "discount_factor": 0.9},
    ]

    results_summary = []

    for cfg in configs:
        print(f"\n🧠 Testing config: {cfg}")
        results = run_all_tests(test_dir="dataset/train")

        rewards = [r["total_reward"] for r in results if "total_reward" in r]
        success = [r["success"] for r in results if "success" in r]

        avg_reward = statistics.mean(rewards)
        success_rate = sum(success) / len(success)
        results_summary.append({**cfg, "avg_reward": avg_reward, "success_rate": success_rate})

        print(f"→ Avg Reward: {avg_reward:.3f}, Success: {success_rate:.1%}")

    # Pick best config
    best = max(results_summary, key=lambda x: x["avg_reward"])
    print("\n🏆 Best config:", best)

    # Validate on val set
    print("\n📊 Validating best config on val set...")
    val_results = run_all_tests(test_dir="dataset/val")

    val_rewards = [r["total_reward"] for r in val_results if "total_reward" in r]
    val_success = [r["success"] for r in val_results if "success" in r]

    print(f"Validation Reward: {statistics.mean(val_rewards):.3f}")
    print(f"Validation Success: {sum(val_success)/len(val_success):.1%}")

    return best
