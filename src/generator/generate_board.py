import json
import random
import uuid


def generate_board(
    width=6,
    height=6,
    wall_density=0.1,
    hole_density=0.05,
    trap_density=0.05,
    coin_density=0.08,
    seed=None,
):
    rng = random.Random(seed)

    def rand_pos():
        return [rng.randint(0, width - 1), rng.randint(0, height - 1)]

    # Ensure agent and goal are different
    agent = rand_pos()
    goal = rand_pos()
    while goal == agent:
        goal = rand_pos()

    def random_positions(count, exclude=None):
        exclude = exclude or set()
        positions = set()
        while len(positions) < count:
            x, y = rand_pos()
            if (x, y) not in exclude:
                positions.add((x, y))
        return [list(p) for p in positions]

    total_cells = width * height
    wall_count = int(total_cells * wall_density)
    hole_count = int(total_cells * hole_density)
    trap_count = int(total_cells * trap_density)
    coin_count = int(total_cells * coin_density)

    occupied = {tuple(agent), tuple(goal)}

    walls = random_positions(wall_count, occupied)
    occupied.update(map(tuple, walls))
    holes = random_positions(hole_count, occupied)
    occupied.update(map(tuple, holes))
    traps = random_positions(trap_count, occupied)
    occupied.update(map(tuple, traps))
    coins = random_positions(coin_count, occupied)
    occupied.update(map(tuple, coins))

    # Use UUID for unique name and filename
    unique_id = str(uuid.uuid4())
    filename = f"board_{unique_id}.json"

    board = {
        "width": width,
        "height": height,
        "name": f"Board_{unique_id}",
        "description": f"Generated automatically with ID {unique_id}.",
        "agent": agent,
        "goal": {
            "position": goal,
            "reward": 10.0,
            "terminal": True
        },
        "walls": {
            "positions": walls,
            "penalty": -5.0
        },
        "holes": {
            "positions": holes,
            "penalty": -10.0,
            "terminal": True
        },
        "traps": {
            "positions": traps,
            "penalty": -8.0,
            "terminal": False
        },
        "coins": {
            "positions": coins,
            "reward": 2.0
        },
        "rewards": {
            "step": -0.1
        },
        "move_probabilities": {
            "intended": 0.8,
            "left_slip": 0.1,
            "right_slip": 0.1
        },
        "observation_noise": {
            "sensor_noise": 0.1,
            "sensor_failure": 0.05
        },
        "solver": "pomcp",
        "solver_config": {
            "max_depth": 20,
            "discount_factor": 0.95,
            "exploration_const": 10.0,
            "num_sims": 1000
        }
    }

    with open(filename, "w") as f:
        json.dump(board, f, indent=4)

    print(f"✅ Board file created: {filename}")
    return filename
