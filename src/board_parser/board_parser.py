import json
from jsonschema import validate, ValidationError

from src.board_parser.board_schema import BOARD_SCHEMA


class BoardParser:
    @staticmethod
    def parse(board_file_path):
        with open(board_file_path, 'r') as f:
            data = json.load(f)
            BoardParser._validate_board_layout(data)
            BoardParser._validate_positions(data)
            return data


    @staticmethod
    def _validate_board_layout( board_layout: dict[str, any]):
        """Validates the board layout"""
        try:
            validate(instance=board_layout, schema=BOARD_SCHEMA)
        except ValidationError as e:
            print("Board JSON validation error:", e.message)
            raise


    @staticmethod
    def _validate_positions(board_layout: dict[str, any]):
        """Ensures no overlapping or duplicate positions, and that goal/agent are valid."""
        def as_tuple_list(key):
            return [tuple(pos) for pos in board_layout.get(key, {}).get("positions", [])]

        categories = ["walls", "holes", "traps", "coins"]
        positions = {name: as_tuple_list(name) for name in categories}

        goal_pos = tuple(board_layout["goal"]["position"])
        agent_pos = tuple(board_layout["agent"])

        # --- Check for duplicates within each category ---
        for name, coords in positions.items():
            duplicates = [pos for pos in set(coords) if coords.count(pos) > 1]
            if duplicates:
                raise ValueError(f"Duplicate positions found in {name}: {duplicates}")

        # --- Check for overlaps between different categories ---
        all_items = list(positions.items())
        for i, (name1, coords1) in enumerate(all_items):
            for name2, coords2 in all_items[i + 1:]:
                overlap = set(coords1) & set(coords2)
                if overlap:
                    raise ValueError(f"Overlapping positions between {name1} and {name2}: {list(overlap)}")

        # --- Check goal position not obstructed ---
        for name, coords in positions.items():
            if goal_pos in coords:
                raise ValueError(f"Goal position {goal_pos} is obstructed by {name}.")

        # --- Check agent not inside wall, hole, or trap ---
        for name in ["walls", "holes", "traps"]:
            if agent_pos in positions[name]:
                raise ValueError(f"Agent starts in invalid position ({agent_pos}) overlapping {name}.")

        print("Board position validation passed.")