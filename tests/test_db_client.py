import pytest
from unittest.mock import AsyncMock, MagicMock
from src.db.client import SupabaseClient


def make_result():
    return {
        "scenario": "baseline",
        "maze_seed": 42,
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "model_version": "claude-sonnet-4-6",
        "run_number": 1,
        "lookahead": 3,
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:01:00+00:00",
        "total_prompt_tokens": 300,
        "total_completion_tokens": 150,
        "total_tokens": 450,
        "observer_tokens": None,
        "agents": [
            {
                "agent_id": str(i + 1),
                "path": [(1, 1), (1, 2), (1, 3)],
                "steps": 2,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "reached_exit": False,
            }
            for i in range(3)
        ],
    }


def make_client() -> SupabaseClient:
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.insert.return_value.execute = AsyncMock(
        return_value=MagicMock(data=[{"id": "test-uuid-123"}])
    )
    return SupabaseClient(mock_supabase)


@pytest.mark.asyncio
async def test_save_run_returns_experiment_id():
    client = make_client()
    experiment_id = await client.save_run(make_result(), maze_id=1)
    assert experiment_id == "test-uuid-123"


@pytest.mark.asyncio
async def test_save_run_observer_tokens_none():
    client = make_client()
    result = make_result()
    result["observer_tokens"] = None

    await client.save_run(result, maze_id=1)

    inserted_row = client._client.table.return_value.insert.call_args_list[0][0][0]
    assert inserted_row["observer_prompt_tokens"] is None
    assert inserted_row["observer_completion_tokens"] is None


@pytest.mark.asyncio
async def test_save_run_with_observer_tokens():
    client = make_client()
    result = make_result()
    result["observer_tokens"] = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}

    await client.save_run(result, maze_id=1)

    inserted_row = client._client.table.return_value.insert.call_args_list[0][0][0]
    assert inserted_row["observer_prompt_tokens"] == 20
    assert inserted_row["observer_completion_tokens"] == 10


@pytest.mark.asyncio
async def test_save_run_inserts_three_tables():
    client = make_client()
    await client.save_run(make_result(), maze_id=1)

    table_calls = [call[0][0] for call in client._client.table.call_args_list]
    assert "experiments" in table_calls
    assert "agent_runs" in table_calls
    assert "trajectories" in table_calls


@pytest.mark.asyncio
async def test_save_run_costs_calculated():
    client = make_client()
    await client.save_run(make_result(), maze_id=1)

    row = client._client.table.return_value.insert.call_args_list[0][0][0]
    # claude-sonnet-4-6: $3/M input, $15/M output
    # 300 prompt → 300 * 3 / 1_000_000 = 0.0009
    # 150 completion → 150 * 15 / 1_000_000 = 0.00225
    assert row["cost_prompt_usd"]     == pytest.approx(0.0009,   rel=1e-4)
    assert row["cost_completion_usd"] == pytest.approx(0.00225,  rel=1e-4)
    assert row["cost_agents_usd"]     == pytest.approx(0.00315,  rel=1e-4)
    assert row["cost_observer_usd"]   is None
    assert row["cost_total_usd"]      == pytest.approx(0.00315,  rel=1e-4)


@pytest.mark.asyncio
async def test_save_run_costs_with_observer():
    client = make_client()
    result = make_result()
    result["observer_tokens"] = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    await client.save_run(result, maze_id=1)

    row = client._client.table.return_value.insert.call_args_list[0][0][0]
    # observer: 100 * 3/1M + 50 * 15/1M = 0.0003 + 0.00075 = 0.00105
    assert row["cost_observer_usd"] == pytest.approx(0.00105, rel=1e-4)
    assert row["cost_total_usd"]    == pytest.approx(0.00315 + 0.00105, rel=1e-4)