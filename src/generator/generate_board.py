import json
import random


def generate_board_from_config(config, seed=None, board_id=None):
    """
    Generate a board based on a configuration dictionary

    Example config:
    {
        "width": 6,
        "height": 6,
        "wall_density": 0.1,
        "hole_density": 0.05,
        "trap_density": 0.05,
        "coin_density": 0.08,
        "goal_reward": 100.0,
        "wall_penalty": -5.0,
        "hole_penalty": -10.0,
        "trap_penalty": -8.0,
        "coin_reward": 20.0,
        "step_reward": -1,
        "move_probabilities": {"intended":0.8,"left_slip":0.1,"right_slip":0.1},
        "observation_noise": {"sensor_noise":0.3,"sensor_failure":0.1},
        "solver":"pomcp",
        "solver_config": {"max_depth":20,"discount_factor":0.95,"exploration_const":100,"num_sims":1000}
    }
    """
    rng = random.Random(seed)

    width = config.get("width", 6)
    height = config.get("height", 6)

    def rand_pos():
        return [rng.randint(0, width - 1), rng.randint(0, height - 1)]

    # Agent and goal
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
    wall_count = int(total_cells * config.get("wall_density", 0.1))
    hole_count = int(total_cells * config.get("hole_density", 0.05))
    trap_count = int(total_cells * config.get("trap_density", 0.05))
    coin_count = int(total_cells * config.get("coin_density", 0.08))

    occupied = {tuple(agent), tuple(goal)}

    walls = random_positions(wall_count, occupied)
    occupied.update(map(tuple, walls))
    holes = random_positions(hole_count, occupied)
    occupied.update(map(tuple, holes))
    traps = random_positions(trap_count, occupied)
    occupied.update(map(tuple, traps))
    coins = random_positions(coin_count, occupied)
    occupied.update(map(tuple, coins))

    if board_id is not None:
        unique_id = board_id
    elif seed is not None:
        unique_id = seed
    else:
        unique_id = 0

    filename = f"board_{unique_id:04d}.json"

    board = {
        "width": width,
        "height": height,
        "name": f"Board_{unique_id}",
        "description": f"Generated automatically with ID {unique_id}.",
        "agent": agent,
        "goal": {
            "position": goal,
            "reward": config.get("goal_reward", 100.0),
            "terminal": True
        },
        "walls": {
            "positions": walls,
            "penalty": config.get("wall_penalty", -5.0)
        },
        "holes": {
            "positions": holes,
            "penalty": config.get("hole_penalty", -10.0),
            "terminal": True
        },
        "traps": {
            "positions": traps,
            "penalty": config.get("trap_penalty", -8.0),
            "terminal": False
        },
        "coins": {
            "positions": coins,
            "reward": config.get("coin_reward", 20.0)
        },
        "rewards": {
            "step": config.get("step_reward", -1)
        },
        "move_probabilities": config.get("move_probabilities", {"intended": 0.8, "left_slip": 0.1, "right_slip": 0.1}),
        "observation_noise": config.get("observation_noise", {"sensor_noise": 0.3, "sensor_failure": 0.1}),
        "solver": config.get("solver", "pouct"),
        "solver_config": config.get("solver_config",
                                    {"max_depth": 20, "discount_factor": 0.95, "exploration_const": 100,
                                     "num_sims": 1000})
    }

    with open(filename, "w") as f:
        json.dump(board, f, indent=4)

    print(f"✅ Board file created: {filename}")
    return filename