import json
from src.agents.context import _filter_recent_flat, _summarize_turn, compress_messages, _extract_insight_summary


# ── helpers ──────────────────────────────────────────────────────────────────

class _Fn:
    def __init__(self, name, args="{}"):
        self.name = name
        self.arguments = args

class _TC:
    def __init__(self, id_, name):
        self.id = id_
        self.function = _Fn(name)

class _Msg:
    """Minimal assistant message with tool_calls (mimics litellm response object)."""
    def __init__(self, tool_calls):
        self.role = "assistant"
        self.tool_calls = tool_calls
        self.content = None

def _tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(content)}

def _system():
    return {"role": "system", "content": "system prompt"}

def _user(text):
    return {"role": "user", "content": text}


# ── _filter_recent_flat ───────────────────────────────────────────────────────

def test_filter_is_passthrough():
    """_filter_recent_flat is now a no-op — all results preserved as-is."""
    asst = _Msg([_TC("m1", "move")])
    tool = _tool_result("m1", {"success": True})
    flat = [asst, tool]
    assert _filter_recent_flat(flat) is flat


# ── _extract_insight_summary ──────────────────────────────────────────────────

_OBSERVER_REPORT = (
    "**OBSERVER REPORT FOR AGENT 1**\n\n"
    "**1. MAP ANALYSIS**\nSome map info.\n\n"
    "**2. DISTANCE ANALYSIS**\nAgent 2 is closest.\n\n"
    "**3. AGENT STATUS**\nAgent 1: (11,3)\n\n"
    "**4. STRATEGIC RECOMMENDATION**\n"
    "Head south to the confirmed waypoint (11,11) first.\n\n"
    "**5. IMMEDIATE NEXT MOVE**\n"
    "Move south to (11,4) — only open direction toward waypoint.\n"
)


def test_extract_insight_summary_sections_4_and_5():
    summary = _extract_insight_summary(_OBSERVER_REPORT)
    assert "rec:" in summary
    assert "waypoint" in summary
    assert "move:" in summary
    assert "south" in summary


def test_extract_insight_summary_missing_sections():
    summary = _extract_insight_summary("No sections here.")
    assert summary == "insight received"


# ── _summarize_turn: get_insight ──────────────────────────────────────────────

def test_summarize_get_insight_includes_rec_and_move():
    tc = _TC("ins1", "get_insight")
    asst = _Msg([tc])
    tool = _tool_result("ins1", {"recommendation": _OBSERVER_REPORT})
    summary = _summarize_turn(asst, [tool])
    assert "get_insight→" in summary
    assert "rec:" in summary
    assert "move:" in summary



# ── _summarize_turn: recent_path in move ─────────────────────────────────────

def test_summarize_move_includes_recent_path():
    tc = _TC("m1", "move")
    tc.function.arguments = json.dumps({"direction": "north"})
    asst = _Msg([tc])
    tool = _tool_result("m1", {
        "success": True,
        "position": {"x": 3, "y": 2},
        "recent_path": [
            {"x": 3, "y": 5, "dist_to_exit": 8},
            {"x": 3, "y": 4, "dist_to_exit": 7},
            {"x": 3, "y": 3, "dist_to_exit": 6},
        ],
    })
    summary = _summarize_turn(asst, [tool])
    assert "recent:[(3,5)→(3,4)→(3,3)]" in summary


def test_summarize_move_without_recent_path():
    tc = _TC("m2", "move")
    tc.function.arguments = json.dumps({"direction": "east"})
    asst = _Msg([tc])
    tool = _tool_result("m2", {"success": False, "position": {"x": 1, "y": 1}})
    summary = _summarize_turn(asst, [tool])
    assert "move(east)->wall pos=(1,1)" in summary
    assert "recent" not in summary


def test_compress_shared_memory_results_preserved():
    """get_shared_memory results in recent turns are kept as-is (no automatic replacement)."""
    sm_tc1 = _TC("sm1", "get_shared_memory")
    sm_tc2 = _TC("sm2", "get_shared_memory")
    messages = [
        _system(),
        _user("Start navigating. Find the exit."),
        _Msg([sm_tc1]),
        _tool_result("sm1", {"agent_2": {"current": {"x": 1, "y": 1}, "visited": []}}),
        _Msg([sm_tc2]),
        _tool_result("sm2", {"agent_2": {"current": {"x": 2, "y": 2}, "visited": []}}),
    ]
    result = compress_messages(messages)
    tool_results = [m for m in result if isinstance(m, dict) and m.get("role") == "tool"]
    contents = [tr["content"] for tr in tool_results]
    assert any("x" in c for c in contents), "shared memory results should be preserved in context"