import pytest
import asyncio
from src.maze.generator import generate_maze
from src.agents.navigator import NavigatorAgent


def make_agent(agent_index=0, seed=42):
    maze = generate_maze(seed=seed)
    return NavigatorAgent(
        agent_id=str(agent_index + 1),
        agent_index=agent_index,
        maze=maze,
        model="claude-sonnet-4-6",
    )


def test_initial_position():
    agent = make_agent(agent_index=0)
    assert agent.position == agent.maze.start_positions[0]


def test_surroundings_returns_four_directions():
    agent = make_agent()
    result = agent._surroundings()
    assert set(result.keys()) == {"north", "south", "east", "west"}


def test_surroundings_values_within_lookahead():
    agent = make_agent()
    result = agent._surroundings()
    for v in result.values():
        assert 0 <= v <= agent.lookahead


@pytest.mark.asyncio
async def test_move_valid_direction():
    agent = make_agent()
    surroundings = agent._surroundings()
    open_dirs = [d for d, v in surroundings.items() if v > 0]
    assert open_dirs, "Agent must have at least one open direction from start"

    start = agent.position
    result = await agent._move(open_dirs[0])
    assert result["success"] is True
    assert agent.position != start
    assert len(agent.path) == 2


@pytest.mark.asyncio
async def test_move_into_wall():
    agent = make_agent()
    surroundings = agent._surroundings()
    wall_dirs = [d for d, v in surroundings.items() if v == 0]

    if not wall_dirs:
        pytest.skip("No wall directions from this start position")

    start = agent.position
    result = await agent._move(wall_dirs[0])
    assert result["success"] is False
    assert agent.position == start
    assert len(agent.path) == 1


def test_trace_entry_has_token_fields():
    agent = make_agent()
    agent.trace.append({
        "step": 0,
        "llm_text": "thinking",
        "tool_name": "get_location",
        "tool_args": {},
        "tool_result": {"x": 1, "y": 1},
        "prompt_tokens": 100,
        "completion_tokens": 20,
    })
    entry = agent.trace[0]
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 20


def test_trace_subsequent_tool_calls_have_none_tokens():
    agent = make_agent()
    # Simulate two tool calls in the same LLM turn
    agent.trace.append({
        "step": 0, "llm_text": "thinking", "tool_name": "get_location",
        "tool_args": {}, "tool_result": {}, "prompt_tokens": 150, "completion_tokens": 30,
    })
    agent.trace.append({
        "step": 1, "llm_text": None, "tool_name": "move",
        "tool_args": {"direction": "south"}, "tool_result": {}, "prompt_tokens": None, "completion_tokens": None,
    })
    assert agent.trace[0]["prompt_tokens"] == 150
    assert agent.trace[1]["prompt_tokens"] is None


@pytest.mark.asyncio
async def test_get_shared_memory_returns_other_agents():
    from src.memory.shared import SharedMemoryStore
    maze = generate_maze(seed=42)
    store = SharedMemoryStore()
    await store.record("2", 5, 3, 1)
    await store.record("2", 6, 3, 2)
    await store.record("3", 9, 7, 1)

    agent = NavigatorAgent(
        agent_id="1", agent_index=0, maze=maze,
        model="claude-sonnet-4-6", shared_memory=store,
    )

    class FakeToolCall:
        class function:
            name = "get_shared_memory"
            arguments = "{}"
        id = "fake"

    result = await agent._execute(FakeToolCall())
    assert "agent_2" in result
    assert "agent_3" in result
    assert "agent_1" not in result  # excludes self
    assert result["agent_2"]["current"] == {"x": 6, "y": 3}
    assert len(result["agent_2"]["visited"]) == 2


@pytest.mark.asyncio
async def test_get_shared_memory_excludes_self():
    from src.memory.shared import SharedMemoryStore
    maze = generate_maze(seed=42)
    store = SharedMemoryStore()
    await store.record("1", 1, 1, 1)
    await store.record("2", 5, 3, 1)

    agent = NavigatorAgent(
        agent_id="1", agent_index=0, maze=maze,
        model="claude-sonnet-4-6", shared_memory=store,
    )

    class FakeToolCall:
        class function:
            name = "get_shared_memory"
            arguments = "{}"
        id = "fake"

    result = await agent._execute(FakeToolCall())
    assert "agent_1" not in result
    assert "agent_2" in result


@pytest.mark.asyncio
async def test_move_writes_to_shared_memory():
    from src.memory.shared import SharedMemoryStore
    maze = generate_maze(seed=42)
    store = SharedMemoryStore()
    agent = NavigatorAgent(
        agent_id="1",
        agent_index=0,
        maze=maze,
        model="claude-sonnet-4-6",
        shared_memory=store,
    )

    surroundings = agent._surroundings()
    open_dirs = [d for d, v in surroundings.items() if v > 0]
    await agent._move(open_dirs[0])

    data = await store.get_all()
    assert "1" in data
    assert len(data["1"]) == 1