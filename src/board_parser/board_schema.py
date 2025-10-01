BOARD_SCHEMA = {
    "type": "object",
    "required": ["width", "height", "agent", "goal", "walls", "holes", "traps",
                 "rewards", "discount", "move_probabilities", "observation_noise"],
    "properties": {
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "name": {"type": "string"},
        "description": {"type": "string"},

        "agent": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 2,
            "maxItems": 2
        },

        "goal": {
            "type": "object",
            "required": ["position", "reward", "terminal"],
            "properties": {
                "position": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 2,
                    "maxItems": 2
                },
                "reward": {"type": "number"},
                "terminal": {"type": "boolean"}
            }
        },

        "walls": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "minItems": 2,
                "maxItems": 2
            }
        },

        "holes": {
            "type": "object",
            "required": ["positions", "penalty", "terminal"],
            "properties": {
                "positions": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "penalty": {"type": "number"},
                "terminal": {"type": "boolean"}
            }
        },

        "traps": {
            "type": "object",
            "required": ["positions", "penalty", "terminal"],
            "properties": {
                "positions": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "penalty": {"type": "number"},
                "terminal": {"type": "boolean"}
            }
        },

        "rewards": {
            "type": "object",
            "required": ["step"],
            "properties": {
                "step": {"type": "number"}
            }
        },

        "discount": {"type": "number", "minimum": 0, "maximum": 1},

        "move_probabilities": {
            "type": "object",
            "required": ["intended", "left_slip", "right_slip"],
            "properties": {
                "intended": {"type": "number", "minimum": 0, "maximum": 1},
                "left_slip": {"type": "number", "minimum": 0, "maximum": 1},
                "right_slip": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },

        "observation_noise": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "additionalProperties": False
}
