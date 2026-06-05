import json
import pytest
from src.agents.context import _filter_recent_flat, compress_messages


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

def test_filter_replaces_shared_memory_result():
    asst = _Msg([_TC("id1", "get_shared_memory")])
    tool = _tool_result("id1", {"agent_2": {"current": {"x": 1, "y": 2}, "visited": []}})
    result = _filter_recent_flat([asst, tool])

    assert result[0] is asst
    assert result[1]["content"] == "[stale — see [CURRENT SHARED MEMORY] below]"
    assert result[1]["tool_call_id"] == "id1"


def test_filter_keeps_latest_insight_removes_older():
    asst1 = _Msg([_TC("ins1", "get_insight")])
    tool1 = _tool_result("ins1", {"recommendation": "go north"})
    asst2 = _Msg([_TC("ins2", "get_insight")])
    tool2 = _tool_result("ins2", {"recommendation": "go east"})

    result = _filter_recent_flat([asst1, tool1, asst2, tool2])

    # older insight replaced, latest kept intact
    assert result[1]["content"] == "[stale — see latest get_insight result below]"
    assert json.loads(result[3]["content"])["recommendation"] == "go east"


def test_filter_passthrough_when_no_ephemeral_tools():
    asst = _Msg([_TC("m1", "move")])
    tool = _tool_result("m1", {"success": True})
    flat = [asst, tool]
    assert _filter_recent_flat(flat) is flat  # same object returned


def test_filter_preserves_move_results_alongside_shared_memory():
    move_tc = _TC("m1", "move")
    mem_tc  = _TC("sm1", "get_shared_memory")
    asst1 = _Msg([move_tc])
    asst2 = _Msg([mem_tc])
    move_result = _tool_result("m1", {"success": True, "position": {"x": 3, "y": 4}})
    mem_result  = _tool_result("sm1", {"agent_2": {}})

    result = _filter_recent_flat([asst1, move_result, asst2, mem_result])

    assert result[1]["content"] == json.dumps({"success": True, "position": {"x": 3, "y": 4}})
    assert result[3]["content"] == "[stale — see [CURRENT SHARED MEMORY] below]"


def test_filter_single_insight_not_replaced():
    asst = _Msg([_TC("ins1", "get_insight")])
    tool = _tool_result("ins1", {"recommendation": "go south"})
    result = _filter_recent_flat([asst, tool])
    assert json.loads(result[1]["content"])["recommendation"] == "go south"


# ── compress_messages with fresh_context ─────────────────────────────────────

def _make_messages_with_shared_memory():
    """Two turns: each calls get_shared_memory."""
    sm_tc1 = _TC("sm1", "get_shared_memory")
    sm_tc2 = _TC("sm2", "get_shared_memory")
    asst1 = _Msg([sm_tc1])
    asst2 = _Msg([sm_tc2])
    return [
        _system(),
        _user("Start navigating. Find the exit."),
        asst1,
        _tool_result("sm1", {"agent_2": {"current": {"x": 1, "y": 1}, "visited": []}}),
        asst2,
        _tool_result("sm2", {"agent_2": {"current": {"x": 2, "y": 2}, "visited": []}}),
    ]


def _injected_fresh(result):
    """Return messages that ARE the fresh injection (start with '[CURRENT SHARED MEMORY] {')."""
    return [m for m in result if isinstance(m, dict) and m.get("content", "").startswith("[CURRENT SHARED MEMORY] {")]


def test_compress_injects_fresh_context_once():
    fresh = {"agent_2": {"current": {"x": 5, "y": 5}, "visited": [{"x": 1, "y": 1}]}}
    result = compress_messages(_make_messages_with_shared_memory(), fresh_context=fresh)

    fresh_msgs = _injected_fresh(result)
    assert len(fresh_msgs) == 1
    injected = json.loads(fresh_msgs[0]["content"].replace("[CURRENT SHARED MEMORY] ", ""))
    assert injected["agent_2"]["current"] == {"x": 5, "y": 5}


def test_compress_no_fresh_context_leaves_no_injection():
    result = compress_messages(_make_messages_with_shared_memory(), fresh_context=None)
    assert _injected_fresh(result) == []


def test_compress_stale_shared_memory_results_replaced():
    fresh = {"agent_2": {"current": {"x": 5, "y": 5}, "visited": []}}
    result = compress_messages(_make_messages_with_shared_memory(), fresh_context=fresh)

    tool_results = [m for m in result if isinstance(m, dict) and m.get("role") == "tool"]
    for tr in tool_results:
        assert tr["content"] == "[stale — see [CURRENT SHARED MEMORY] below]"