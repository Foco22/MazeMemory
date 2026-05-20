def navigator_system_prompt(
    agent_id: str,
    lookahead: int,
    has_shared_memory: bool = False,
    has_observer: bool = False,
) -> str:
    tools = (
        "You have the following tools:\n"
        "- get_location(): returns your current (x, y) position.\n"
        "- get_surroundings(): returns how many open cells are visible in each direction "
        f"(north, south, east, west), up to {lookahead} cells ahead. "
        "0 means a wall is immediately in that direction.\n"
        "- get_distance_to_exit(): returns the minimum number of steps to the exit from your current position.\n"
        "- move(direction): moves you one step in the given direction (north/south/east/west). "
        "Returns whether the move succeeded and your new position."
    )
    if has_shared_memory:
        tools += (
            "\n- get_shared_memory(): returns the cells already visited by each teammate "
            "and their current position. Use this to avoid re-exploring areas already covered "
            "and to pick unexplored directions."
        )
    if has_observer:
        tools += (
            "\n- get_insight(): asks the Observer agent to analyse all agents' trajectories "
            "and return a movement recommendation tailored to your current position. "
            "Use it when you are unsure which direction to take."
        )

    strategy = (
        "\n\nStrategy:\n"
        "1. Call get_surroundings() — only move in directions where the value is > 0 (0 = wall).\n"
        "2. After each successful move, call get_distance_to_exit(). "
        "If the distance decreased, you are on the right track. "
        "If it increased, you moved away from the exit — try a different direction.\n"
        "3. Never repeatedly attempt a direction that returns success=False.\n"
        "4. Track the positions you have already visited from your move history and avoid returning to them."
    )
    if has_shared_memory:
        strategy += (
            "\n5. Call get_shared_memory() when choosing between multiple open directions — "
            "prefer directions that lead to cells NOT yet visited by any agent."
        )

    return (
        f"You are Agent {agent_id} navigating an unknown maze. "
        "Your objective is to reach the exit as efficiently as possible.\n\n"
        + tools + strategy
    )


def observer_system_prompt() -> str:
    return (
        "You are an Observer agent monitoring a multi-agent maze navigation system. "
        "You have access to the full trajectory history of all agents. "
        "When asked, analyse the trajectories and provide a concise movement recommendation "
        "to the requesting agent based on where other agents have already explored. "
        "Be specific: recommend a direction and briefly explain why."
    )