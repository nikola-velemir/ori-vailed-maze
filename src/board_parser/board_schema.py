BOARD_SCHEMA = {
    "type": "object",
    "required": [
        "width", "height", "agent", "goal", "walls", "holes", "traps", "coins",
        "rewards", "move_probabilities", "observation_noise",
    ],
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
            "type": "object",
            "required": ["positions", "penalty"],
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
            },
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
                        "minItems": 0,
                        "maxItems": 2
                    }
                },
                "penalty": {"type": "number"},
                "terminal": {"type": "boolean"}
            }
        },
        "coins":{
            'type': 'object',
            "required":["positions","reward"],
            "properties": {
                "positions": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 0,
                        "maxItems": 2
                    }
                },
                "reward": {"type": "number"},
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
                        "minItems": 0,
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

        "move_probabilities": {
            "type": "object",
            "required": ["intended", "left_slip", "right_slip"],
            "properties": {
                "intended": {"type": "number", "minimum": 0, "maximum": 1},
                "left_slip": {"type": "number", "minimum": 0, "maximum": 1},
                "right_slip": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },

        "observation_noise": {
            "type": "object",
            "required": ["sensor_noise", "sensor_failure"],
            "properties": {
                "sensor_noise": {"type": "number", "minimum": 0, "maximum": 1},
                "sensor_failure": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "solver": {
            "type": "string",
            "enum": ["pomcp", "pouct"],
            "description": "The solver (planner) used for POMDP solving."
        },

        "solver_config": {
            "type": "object",
            "properties": {
                "max_depth": {"type": "integer", "minimum": 1},
                "discount_factor": {"type": "number", "minimum": 0, "maximum": 1},
                "exploration_const": {"type": "number", "minimum": 0},
                "num_sims": {"type": "integer", "minimum": 1}
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}
