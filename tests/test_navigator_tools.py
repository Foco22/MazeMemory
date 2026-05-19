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