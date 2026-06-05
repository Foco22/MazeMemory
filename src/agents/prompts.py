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
        "Returns whether the move succeeded, your new position, and recent_path — "
        "the last 5 cells you visited with their distance to exit. "
        "Use recent_path to avoid moving back to cells you just came from."
    )
    if has_shared_memory:
        tools += (
            "\n- get_shared_memory(): returns the cells already visited by each teammate "
            "and their current position. Use this to avoid re-exploring areas already covered "
            "and to pick unexplored directions. "
            "You also receive a [CURRENT SHARED MEMORY] message at the end of each turn with the "
            "latest teammate state — it is always up to date and can be used directly."
        )
    if has_observer:
        tools += (
            f"\n- get_insight(): asks the Observer agent to analyse all agents' trajectories "
            f"and return a 4-section report with a movement recommendation. "
            f"Available once every {insight_interval} moves — if called too soon you will receive "
            f"a cooldown message telling you how many moves to wait."
        )

    strategy = "\n\nStrategy:\n"
    next_step = 1
    if has_observer:
        strategy += (
            f"{next_step}. Before moving, check if get_insight() is available. "
            f"It becomes available every {insight_interval} successful moves. "
            "If available, call it: use section 5 (IMMEDIATE NEXT MOVE) to choose your next direction, "
            "and section 4 (STRATEGIC RECOMMENDATION) to understand the long-term goal. "
            "If on cooldown, skip and proceed with the steps below.\n"
        )
        next_step += 1
    strategy += (
        f"{next_step}. Call get_surroundings() — only move in directions where the value is > 0 (0 = wall).\n"
    )
    next_step += 1
    strategy += (
        f"{next_step}. After each successful move, call get_distance_to_exit(). "
        "If the distance decreased, you are heading toward the exit — continue in that direction. "
        "If the distance increased, you moved away — immediately try a different open direction.\n"
    )
    next_step += 1
    strategy += (
        f"{next_step}. Never repeatedly attempt a direction that returns success=False.\n"
    )
    next_step += 1
    strategy += (
        f"{next_step}. Each move() response includes recent_path — the last 5 cells you visited. "
        "Avoid directions that lead back into recent_path to prevent going in circles."
    )
    next_step += 1
    if has_shared_memory:
        strategy += (
            f"\n{next_step}. When choosing between multiple open directions, "
            "consult [CURRENT SHARED MEMORY] and prefer directions toward cells "
            "NOT yet visited by any teammate."
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
        "Respond with a structured report in exactly 5 sections:\n\n"
        "1. MAP ANALYSIS\n"
        "   Describe which zones are already explored and which corridors remain unexplored. "
        "Be specific about coordinates or regions.\n\n"
        "2. DISTANCE ANALYSIS\n"
        "   Identify which agent is closest to the exit. "
        "Explain what this implies for the other agents' coordination "
        "(e.g. should they converge, spread out, or follow a specific route).\n\n"
        "3. AGENT STATUS\n"
        "   One line per agent: current position, distance to exit, steps taken so far.\n\n"
        "4. STRATEGIC RECOMMENDATION\n"
        "   The overall direction the requesting agent should head and why — "
        "based on where the exit is or where unexplored zones are. One sentence. "
        "This is the long-term goal.\n\n"
        "5. IMMEDIATE NEXT MOVE\n"
        "   Looking at the requesting agent's current position on the map, identify which "
        "adjacent directions (north/south/east/west) have open corridors (· or E, not █). "
        "From those, recommend the 1-2 directions most likely to reduce distance to the exit "
        "in the next few steps — even if that means going north or west temporarily to get "
        "around a wall. Be specific about which direction(s) to try first. One or two sentences."
    )