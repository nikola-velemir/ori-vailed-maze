import os
import json
import hashlib

from src.generator.generate_board import generate_board_from_config


def generate_dataset_split(
        train_size=60,
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

    with open("config.json", "r") as f:
        config = json.load(f)

    for split, count in splits.items():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"\n📁 Generating {count} boards for {split}/ ...")

        created_files = []

        for i in range(count):
            split_hash = int(hashlib.md5(split.encode()).hexdigest(), 16) % 1000
            board_seed = seed + i + split_hash

            prev_dir = os.getcwd()
            os.chdir(split_dir)

            try:
                filename = generate_board_from_config(config, seed=board_seed, board_id=i)
            finally:
                os.chdir(prev_dir)

            created_files.append(os.path.basename(filename))
            print(f"  ✅ [{i + 1}/{count}] {os.path.basename(filename)}")

        index[split] = created_files

    # Save index file
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=4)

    print("\n📦 Dataset generation complete!")
    print(f"📄 Index file saved to: {index_path}")


if __name__ == "__main__":
    generate_dataset_split(train_size=60, val_size=20, test_size=20)