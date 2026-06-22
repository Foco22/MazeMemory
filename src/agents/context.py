import json
import os
import re

WINDOW_TURNS   = int(os.getenv("CONTEXT_WINDOW_TURNS", 10))
FULL_TURNS     = int(os.getenv("CONTEXT_FULL_TURNS", 2))
COMPRESS_TURNS = WINDOW_TURNS - FULL_TURNS


def _tc_name(tc) -> str:
    return tc["function"]["name"] if isinstance(tc, dict) else tc.function.name

def _tc_arguments(tc) -> str:
    return tc["function"]["arguments"] if isinstance(tc, dict) else tc.function.arguments

def _tc_id(tc) -> str:
    return tc["id"] if isinstance(tc, dict) else tc.id


def print_messages(messages: list, label: str = "") -> None:
    print(f"\n{'─'*50} {label}")
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "?")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        tool_calls = None if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        tool_call_id = msg.get("tool_call_id") if isinstance(msg, dict) else None

        if tool_calls:
            calls = ", ".join(f"{_tc_name(tc)}({_tc_arguments(tc)})" for tc in tool_calls)
            print(f"  [{i}] {role} → calls: {calls}")
        elif tool_call_id:
            print(f"  [{i}] tool result: {content}")
        else:
            text = str(content or "")[:200]
            print(f"  [{i}] {role}: {text}")
    print(f"{'─'*50}\n")


def _extract_insight_summary(text: str) -> str:
    """Extract sections 4 (Strategic Recommendation) and 5 (Immediate Next Move) from observer report."""
    rec = move = ""
    m4 = re.search(r'4\.\s+(?:STRATEGIC\s+)?RECOMMENDATION[^\n]*\n+\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    m5 = re.search(r'5\.\s+IMMEDIATE\s+NEXT\s+MOVE[^\n]*\n+\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m4:
        rec = re.sub(r'\*+', '', m4.group(1)).strip()
    if m5:
        move = re.sub(r'\*+', '', m5.group(1)).strip()
    parts = []
    if rec:
        parts.append(f"rec:{rec[:100]}")
    if move:
        parts.append(f"move:{move[:80]}")
    return " | ".join(parts) if parts else "insight received"


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
        name = _tc_name(tc)
        args = json.loads(_tc_arguments(tc) or "{}")
        res  = by_id.get(_tc_id(tc), {})
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
        elif name == "get_insight":
            rec_text = res.get("recommendation", "") if isinstance(res, dict) else str(res)
            parts.append("get_insight→" + _extract_insight_summary(rec_text))
        elif name == "get_shared_memory":
            pass  # raw memory data omitted from compressed history
        else:
            parts.append(f"{name}->{res}")
    return "[step] " + " | ".join(parts)


def _filter_recent_flat(flat: list) -> list:
    return flat


def compress_messages(messages: list) -> list:
    """Sliding window of WINDOW_TURNS turns: last FULL_TURNS full, previous COMPRESS_TURNS compressed, rest discarded."""
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

    return header + compressed + recent_flat
