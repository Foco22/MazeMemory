import json
import os

WINDOW_TURNS   = int(os.getenv("CONTEXT_WINDOW_TURNS", 10))
FULL_TURNS     = int(os.getenv("CONTEXT_FULL_TURNS", 2))
COMPRESS_TURNS = WINDOW_TURNS - FULL_TURNS


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
            parts.append(f"dist_to_exit={res.get('steps_to_exit')}")
        elif name == "move":
            d  = args.get("direction", "?")
            ok = res.get("success")
            p  = res.get("position", {})
            recent = res.get("recent_path", [])
            recent_str = "→".join(f"({r['x']},{r['y']})" for r in recent)
            summary = f"move({d})->{'ok' if ok else 'wall'} pos=({p.get('x')},{p.get('y')})"
            if recent_str:
                summary += f" recent:[{recent_str}]"
            parts.append(summary)
        elif name in ("get_shared_memory", "get_insight"):
            pass  # omitted from compressed history — fresh state is injected separately
        else:
            parts.append(f"{name}->{res}")
    return "[step] " + " | ".join(parts)


def _filter_recent_flat(flat: list) -> list:
    """
    Replace stale get_shared_memory results with a placeholder (fresh state injected separately).
    Keep only the latest get_insight result; replace older ones with a placeholder.
    """
    shared_mem_ids: set[str] = set()
    insight_ids: list[str] = []

    for msg in flat:
        tool_calls = getattr(msg, "tool_calls", None) if not isinstance(msg, dict) else None
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = getattr(tc.function, "name", "") if hasattr(tc, "function") else ""
            if name == "get_shared_memory":
                shared_mem_ids.add(tc.id)
            elif name == "get_insight":
                insight_ids.append(tc.id)

    stale_insight_ids = set(insight_ids[:-1]) if len(insight_ids) > 1 else set()

    if not shared_mem_ids and not stale_insight_ids:
        return flat

    result = []
    for msg in flat:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tid = msg.get("tool_call_id")
            if tid in shared_mem_ids:
                result.append({**msg, "content": "[stale — see [CURRENT SHARED MEMORY] below]"})
                continue
            if tid in stale_insight_ids:
                result.append({**msg, "content": "[stale — see latest get_insight result below]"})
                continue
        result.append(msg)

    return result


def compress_messages(
    messages: list,
    fresh_context: dict | None = None,
    last_insight: str | None = None,
    last_insight_age: int | None = None,
) -> list:
    """Sliding window of WINDOW_TURNS turns: last FULL_TURNS full, previous COMPRESS_TURNS compressed, rest discarded.

    fresh_context: if provided, injected as [CURRENT SHARED MEMORY] after recent turns.
    last_insight: if provided, injected as [LAST OBSERVER INSIGHT] after shared memory.
    last_insight_age: moves elapsed since the insight was generated — shown in the label.
    """
    header = messages[:2]  # system + "Start navigating"
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

    if len(turns) <= WINDOW_TURNS:
        # not enough turns to trigger sliding — keep all, compress old ones
        old_turns    = turns[:-FULL_TURNS] if len(turns) > FULL_TURNS else []
        recent_turns = turns[-FULL_TURNS:]
    else:
        old_turns    = turns[-WINDOW_TURNS:-FULL_TURNS]   # the 8 to compress
        recent_turns = turns[-FULL_TURNS:]                # the 2 full

    compressed = []
    for turn in old_turns:
        assistant_msg = turn[0]
        tool_results  = [m for m in turn[1:] if isinstance(m, dict) and m.get("role") == "tool"]
        summary = _summarize_turn(assistant_msg, tool_results)
        if summary:
            compressed.append({"role": "user", "content": summary})

    recent_flat = _filter_recent_flat([msg for turn in recent_turns for msg in turn])

    insight_msgs = []
    if last_insight:
        age_str = f" — {last_insight_age} move{'s' if last_insight_age != 1 else ''} ago" if last_insight_age is not None else ""
        insight_msgs = [{"role": "user", "content": f"[LAST OBSERVER INSIGHT{age_str}]\n{last_insight}"}]

    fresh_msgs = []
    if fresh_context:
        fresh_msgs = [{"role": "user", "content": "[CURRENT SHARED MEMORY] " + json.dumps(fresh_context)}]

    return header + compressed + recent_flat + fresh_msgs + insight_msgs
