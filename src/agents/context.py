import json

COMPRESS_KEEP_TURNS = 2

def print_messages(messages: list, label: str = "") -> None:
    print(f"\n{'─'*50} {label}")
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "?")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        tool_calls = None if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        tool_call_id = msg.get("tool_call_id") if isinstance(msg, dict) else None

        if tool_calls:
            calls = ", ".join(f"{tc.function.name}({tc.function.arguments})" for tc in tool_calls)
            print(f"  [{i}] {role} → calls: {calls}")
        elif tool_call_id:
            print(f"  [{i}] tool result: {content}")
        else:
            text = str(content or "")[:200]
            print(f"  [{i}] {role}: {text}")
    print(f"{'─'*50}\n")


def _summarize_turn(assistant_msg, tool_results: list[dict]) -> str:
    tool_calls = getattr(assistant_msg, "tool_calls", None)
    if not tool_calls:
        return ""

    by_id = {}
    for r in tool_results:
        try:
            by_id[r["tool_call_id"]] = json.loads(r["content"])
        except Exception:
            by_id[r["tool_call_id"]] = r["content"]

    parts = []
    for tc in tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")
        res  = by_id.get(tc.id, {})
        if name == "get_location":
            parts.append(f"loc=({res.get('x')},{res.get('y')})")
        elif name == "get_surroundings":
            parts.append(f"surr={{N:{res.get('north')},S:{res.get('south')},E:{res.get('east')},W:{res.get('west')}}}")
        elif name == "get_distance_to_exit":
            parts.append(f"dist={res.get('steps')}")
        elif name == "move":
            d  = args.get("direction", "?")
            ok = res.get("success")
            p  = res.get("position", {})
            parts.append(f"move({d})->{'ok' if ok else 'wall'} pos=({p.get('x')},{p.get('y')})")
        else:
            parts.append(f"{name}->{res}")
    return "[step] " + " | ".join(parts)


def compress_messages(messages: list, keep_turns: int = COMPRESS_KEEP_TURNS) -> list:
    """Replace old turns with compact text, keeping recent turns in full format."""
    header = messages[:2]  # system + user
    rest   = messages[2:]

    turns: list[list] = []
    current: list = []
    for msg in rest:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "assistant" and current:
            turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)

    if len(turns) <= keep_turns:
        return messages

    old_turns    = turns[:-keep_turns]
    recent_turns = turns[-keep_turns:]

    compressed = []
    for turn in old_turns:
        assistant_msg = turn[0]
        tool_results  = [m for m in turn[1:] if isinstance(m, dict) and m.get("role") == "tool"]
        summary = _summarize_turn(assistant_msg, tool_results)
        if summary:
            compressed.append({"role": "user", "content": summary})

    recent_flat = [msg for turn in recent_turns for msg in turn]
    return header + compressed + recent_flat