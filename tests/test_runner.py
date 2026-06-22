import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.maze.generator import generate_maze
from src.scenarios.runner import run_scenario, SCENARIOS
from src.scenarios.config import ModelConfig

MODEL = ModelConfig(provider="anthropic", model="claude-sonnet-4-6", version="claude-sonnet-4-6")


def make_agent_result(agent_id):
    return {
        "agent_id": agent_id,
        "path": [(1, 1), (1, 2)],
        "steps": 1,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "reached_exit": False,
    }


@pytest.fixture
def maze():
    return generate_maze(seed=42)


@pytest.mark.asyncio
async def test_baseline_no_shared_memory(maze):
    mock_results = [make_agent_result(str(i)) for i in range(3)]

    with patch("src.scenarios.runner.NavigatorAgent") as MockNav:
        instance = AsyncMock()
        instance.run = AsyncMock(side_effect=mock_results)
        MockNav.return_value = instance

        result = await run_scenario("baseline", maze, MODEL, run_number=1)

    assert result["scenario"] == "baseline"
    assert result["observer_tokens"] is None
    # shared_memory must be None for baseline — verify NavigatorAgent was called with it
    call_kwargs = MockNav.call_args_list[0].kwargs
    assert call_kwargs["shared_memory"] is None
    assert call_kwargs["observer"] is None


@pytest.mark.asyncio
async def test_shared_memory_passes_store(maze):
    mock_results = [make_agent_result(str(i)) for i in range(3)]

    with patch("src.scenarios.runner.NavigatorAgent") as MockNav:
        instance = AsyncMock()
        instance.run = AsyncMock(side_effect=mock_results)
        MockNav.return_value = instance

        result = await run_scenario("shared_memory", maze, MODEL, run_number=1)

    assert result["observer_tokens"] is None
    call_kwargs = MockNav.call_args_list[0].kwargs
    assert call_kwargs["shared_memory"] is not None
    assert call_kwargs["observer"] is None


@pytest.mark.asyncio
async def test_shared_memory_observer_includes_observer(maze):
    mock_results = [make_agent_result(str(i)) for i in range(3)]

    with patch("src.scenarios.runner.NavigatorAgent") as MockNav, \
         patch("src.scenarios.runner.ObserverAgent") as MockObs:

        mock_obs_instance = MagicMock()
        mock_obs_instance.prompt_tokens = 20
        mock_obs_instance.completion_tokens = 10
        MockObs.return_value = mock_obs_instance

        nav_instance = AsyncMock()
        nav_instance.run = AsyncMock(side_effect=mock_results)
        MockNav.return_value = nav_instance

        result = await run_scenario("shared_memory_observer", maze, MODEL, run_number=1)

    assert result["observer_tokens"] is not None
    assert result["observer_tokens"]["total_tokens"] == 30
    call_kwargs = MockNav.call_args_list[0].kwargs
    assert call_kwargs["observer"] is mock_obs_instance


@pytest.mark.asyncio
async def test_total_tokens_aggregated(maze):
    mock_results = [make_agent_result(str(i)) for i in range(3)]

    with patch("src.scenarios.runner.NavigatorAgent") as MockNav:
        instance = AsyncMock()
        instance.run = AsyncMock(side_effect=mock_results)
        MockNav.return_value = instance

        result = await run_scenario("baseline", maze, MODEL, run_number=1)

    assert result["total_prompt_tokens"] == 300    # 100 × 3
    assert result["total_completion_tokens"] == 150  # 50 × 3
    assert result["total_tokens"] == 450


@pytest.mark.asyncio
async def test_observer_scenario_no_raw_memory_tool(maze):
    """Observer-only: navigators have shared_memory (to write) and observer,
    but raw_memory_tool=False so they cannot call get_shared_memory directly."""
    mock_results = [make_agent_result(str(i)) for i in range(3)]

    with patch("src.scenarios.runner.NavigatorAgent") as MockNav, \
         patch("src.scenarios.runner.ObserverAgent") as MockObs:

        mock_obs_instance = MagicMock()
        mock_obs_instance.prompt_tokens = 10
        mock_obs_instance.completion_tokens = 5
        MockObs.return_value = mock_obs_instance

        nav_instance = AsyncMock()
        nav_instance.run = AsyncMock(side_effect=mock_results)
        MockNav.return_value = nav_instance

        await run_scenario("observer", maze, MODEL, run_number=1)

    call_kwargs = MockNav.call_args_list[0].kwargs
    assert call_kwargs["shared_memory"] is not None,  "shared_memory must exist so navigators can write"
    assert call_kwargs["observer"] is mock_obs_instance, "observer must be set"
    assert call_kwargs["raw_memory_tool"] is False,   "navigators must not have direct raw memory access"


def test_invalid_scenario_raises(maze):
    import asyncio
    with pytest.raises(AssertionError):
        asyncio.run(run_scenario("invalid", maze, MODEL, run_number=1))