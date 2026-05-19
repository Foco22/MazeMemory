def build_insight_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_insight",
            "description": (
                "Ask the Observer agent to analyse all agents' trajectories and return "
                "a movement recommendation tailored to your current position. "
                "Use when unsure which direction to take."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def build_tools(lookahead: int) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_location",
                "description": "Returns your current (x, y) position in the maze.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_distance_to_exit",
                "description": "Returns the minimum number of steps to the exit from your current position.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_surroundings",
                "description": (
                    f"Returns how many consecutive open cells are visible in each direction "
                    f"(up to {lookahead}). 0 means a wall is immediately in that direction."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move one step in the given direction. Returns whether the move succeeded and your new position.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["north", "south", "east", "west"],
                        }
                    },
                    "required": ["direction"],
                },
            },
        },
    ]