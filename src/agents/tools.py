def build_shared_memory_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_shared_memory",
            "description": (
                "Returns the positions already visited by each teammate and their current position. "
                "Use this to avoid exploring cells already covered by other agents and to coordinate routes."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def build_insight_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_insight",
            "description": (
                "Ask the Observer agent for a navigation report. Returns a 4-section report: "
                "map analysis, distance analysis, agent status, and a specific direction recommendation. "
                "Call this every time it becomes available — do not skip it."
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