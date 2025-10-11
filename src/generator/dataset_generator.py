import os
import json
from generate_board import generate_board


def generate_dataset_split(
    train_size=100,
    val_size=20,
    test_size=20,
    width=8,
    height=8,
    seed=42,
    output_dir="dataset",
):
    """Generate a dataset split into train / val / test folders."""
    splits = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }

    os.makedirs(output_dir, exist_ok=True)
    index = {}

    for split, count in splits.items():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"\n📁 Generating {count} boards for {split}/ ...")

        created_files = []
        for i in range(count):
            # Use different seed offsets for reproducibility
            board_seed = seed + i + (hash(split) % 1000)
            prev_dir = os.getcwd()
            os.chdir(split_dir)
            filename = generate_board(width=width, height=height, seed=board_seed)
            os.chdir(prev_dir)
            created_files.append(filename)
            print(f"  ✅ [{i+1}/{count}] {filename}")

        index[split] = created_files

    # Save index file
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=4)

    print("\n📦 Dataset generation complete!")
    print(f"📄 Index file saved to: {index_path}")


if __name__ == "__main__":
    # Example: generate 100 train, 20 val, 20 test boards
    generate_dataset_split(train_size=80, val_size=20, test_size=20)
