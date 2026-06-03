def navigator_system_prompt(
    agent_id: str,
    lookahead: int,
    has_shared_memory: bool = False,
    has_observer: bool = False,
    insight_interval: int = 7,
) -> str:
    tools = (
        "You have the following tools:\n"
        "- get_location(): returns your current (x, y) position and recent_path — "
        "the last 5 cells you visited with their distance to exit. "
        "Use recent_path to avoid going back to cells you already visited.\n"
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
            f"\n- get_insight(): asks the Observer agent to analyse all agents' trajectories "
            f"and return a 4-section report with a movement recommendation. "
            f"Available once every {insight_interval} moves — if called too soon you will receive "
            f"a cooldown message telling you how many moves to wait."
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
    next_step = 5
    if has_shared_memory:
        strategy += (
            f"\n{next_step}. Call get_shared_memory() when choosing between multiple open directions — "
            "prefer directions that lead to cells NOT yet visited by any agent."
        )
        next_step += 1
    if has_observer:
        strategy += (
            f"\n{next_step}. Call get_insight() every {insight_interval} moves — it is mandatory, not optional. "
            f"As soon as it becomes available (every {insight_interval} successful moves) call it "
            "and follow the recommendation in section 4 of its report."
        )

    return (
        f"You are Agent {agent_id} navigating an unknown maze. "
        "Your objective is to reach the exit as efficiently as possible.\n\n"
        + tools + strategy
    )


def observer_system_prompt() -> str:
    return (
        "You are an Observer agent monitoring a multi-agent maze navigation system.\n"
        "You receive a visual map (█=wall, ·=explored, E=exit, 1/2/3=agent positions), "
        "per-agent status data, and a request from one agent.\n\n"
        "Respond with a structured report in exactly 4 sections:\n\n"
        "1. MAP ANALYSIS\n"
        "   Describe which zones are already explored and which corridors remain unexplored. "
        "Be specific about coordinates or regions.\n\n"
        "2. DISTANCE ANALYSIS\n"
        "   Identify which agent is closest to the exit. "
        "Explain what this implies for the other agents' coordination "
        "(e.g. should they converge, spread out, or follow a specific route).\n\n"
        "3. AGENT STATUS\n"
        "   One line per agent: current position, distance to exit, steps taken so far.\n\n"
        "4. RECOMMENDATION\n"
        "   Give the requesting agent a specific direction (north/south/east/west) "
        "and a one-sentence justification. Prioritise unexplored areas and efficient coordination."
    )