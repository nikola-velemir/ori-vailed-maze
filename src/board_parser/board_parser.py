import json
from jsonschema import validate, ValidationError

from src.board_parser.board_schema import BOARD_SCHEMA


class BoardParser:
    @staticmethod
    def parse(board_file_path):
        with open(board_file_path, 'r') as f:
            data = json.load(f)
            BoardParser._validate_board_layout(data)
            print(data)
            return data


    @staticmethod
    def _validate_board_layout( board_layout: dict[str, any]):
        try:
            validate(instance=board_layout, schema=BOARD_SCHEMA)
        except ValidationError as e:
            print("Board JSON validation error:", e.message)
            raise

