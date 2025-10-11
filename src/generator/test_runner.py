import os
import json
from src.board_parser.board_parser import BoardParser
from src.visuals.runner import run_pomdp_simulation


def run_test_on_board(board_path):
    """Runs one board simulation and returns performance metrics."""
    try:
        board_data = BoardParser.parse(board_file_path=board_path)
        # Create a mock object with the minimal interface expected by run_pomdp_simulation
        class DummyGame:
            def __init__(self, board_data):
                self.board_data = board_data
                self.gamma = board_data["solver_config"]["discount_factor"]

            def get_rewards(self):
                return {
                    'hole_penalty': board_data['holes']['penalty'],
                    'trap_penalty': board_data['traps']['penalty'],
                    'step_cost': board_data['rewards']['step'],
                    'wall_penalty': board_data['walls']['penalty'],
                    'goal_reward': board_data['goal']['reward'],
                    'coin_reward': board_data['coins']['reward'],
                }

            def get_observation_noise(self):
                return board_data["observation_noise"]

            def get_move_probabilites(self):
                return board_data["move_probabilities"]

        game = DummyGame(board_data)

        result = run_pomdp_simulation(game, headless=True)
        # Expect run_pomdp_simulation to return performance metrics like total_reward, steps, success, etc.
        return {"board": os.path.basename(board_path), **result}

    except Exception as e:
        print(f"❌ Error testing {board_path}: {e}")
        return {"board": os.path.basename(board_path), "error": str(e)}


def run_all_tests(test_dir="dataset/test", output_file="results/test_results.json"):
    """Run POMDP simulations on all test boards and save results."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = []

    test_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(".json")
    ]
    print(f"🧩 Found {len(test_files)} test boards in {test_dir}")

    for i, board_path in enumerate(test_files, 1):
        print(f"\n▶️ Running test {i}/{len(test_files)}: {board_path}")
        res = run_test_on_board(board_path)
        results.append(res)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Testing complete! Results saved to {output_file}")
    return results


if __name__ == "__main__":
    run_all_tests()
