from src.metrics.calculator import agent_cost_usd, build_run_summary_rows


def test_agent_cost_usd_with_cache_hit_and_miss():
    # gpt-4o-mini: input=0.15, output=0.60, cache_hit=0.075 ($/1M tokens)
    agent = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "cache_hit_tokens": 600,
        "cache_miss_tokens": 400,
    }
    cost = agent_cost_usd("gpt-4o-mini", agent)
    expected = (600 * 0.075 + 400 * 0.15 + 200 * 0.60) / 1_000_000
    assert cost == expected


def test_agent_cost_usd_without_cache_info():
    # No cache_hit_tokens/cache_miss_tokens present -> all prompt tokens billed at full input price
    agent = {"prompt_tokens": 1000, "completion_tokens": 200}
    cost = agent_cost_usd("gpt-4o-mini", agent)
    expected = (1000 * 0.15 + 200 * 0.60) / 1_000_000
    assert cost == expected


def test_agent_cost_usd_model_without_cache_hit_price():
    # gemini-2.0-flash has no "cache_hit" price -> falls back to input price
    agent = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "cache_hit_tokens": 300,
        "cache_miss_tokens": 700,
    }
    cost = agent_cost_usd("gemini/gemini-2.0-flash", agent)
    expected = (1000 * 0.075 + 200 * 0.30) / 1_000_000
    assert cost == expected


def test_agent_cost_usd_unknown_model_is_zero():
    agent = {"prompt_tokens": 1000, "completion_tokens": 200}
    assert agent_cost_usd("some-unlisted-model", agent) == 0.0


def _make_run_result():
    return {
        "scenario": "baseline",
        "maze_id": 1,
        "run_number": 1,
        "model": "gpt-4o-mini",
        "agents": [
            {
                "agent_id": "1", "steps": 8, "optimal_steps": 8, "optimality_ratio": 1.0,
                "redundancy_ratio": 0.0, "total_tokens": 21201,
                "prompt_tokens": 18000, "completion_tokens": 3201,
            },
            {
                "agent_id": "2", "steps": 8, "optimal_steps": 8, "optimality_ratio": 1.0,
                "redundancy_ratio": 0.0, "total_tokens": 21358,
                "prompt_tokens": 18100, "completion_tokens": 3258,
            },
            {
                "agent_id": "3", "steps": 4, "optimal_steps": 4, "optimality_ratio": 1.0,
                "redundancy_ratio": 0.0, "total_tokens": 9493,
                "prompt_tokens": 8000, "completion_tokens": 1493,
            },
        ],
    }


def test_build_run_summary_rows_has_one_row_per_agent_plus_total():
    rows = build_run_summary_rows(_make_run_result())
    assert [r["Agent"] for r in rows] == ["Agent 1", "Agent 2", "Agent 3", "TOTAL"]


def test_build_run_summary_rows_agent_row_fields():
    rows = build_run_summary_rows(_make_run_result())
    agent1 = rows[0]
    assert agent1["Steps"] == 8
    assert agent1["Optimal"] == 8
    assert agent1["Ratio"] == 1.0
    assert agent1["Redund."] == 0.0
    assert agent1["Tokens"] == 21201
    assert agent1["Cost"] == agent_cost_usd("gpt-4o-mini", _make_run_result()["agents"][0])


def test_build_run_summary_rows_total_sums_tokens_and_cost():
    result = _make_run_result()
    rows = build_run_summary_rows(result)
    total = rows[-1]
    expected_tokens = sum(a["total_tokens"] for a in result["agents"])
    expected_cost = sum(agent_cost_usd(result["model"], a) for a in result["agents"])
    assert total["Tokens"] == expected_tokens
    assert total["Cost"] == expected_cost
    assert total["Steps"] is None
    assert total["Optimal"] is None
