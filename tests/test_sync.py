import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from experiments.sync import sync


def make_result(scenario="baseline", maze_id=1, run_number=1, started_at="2026-05-20T10:00:00+00:00"):
    return {
        "scenario": scenario,
        "maze_id": maze_id,
        "run_number": run_number,
        "started_at": started_at,
        "maze_seed": 42,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "model_version": "gpt-4o-mini",
        "lookahead": 3,
        "completed_at": "2026-05-20T10:00:10+00:00",
        "total_prompt_tokens": 100,
        "total_completion_tokens": 20,
        "total_tokens": 120,
        "observer_tokens": None,
        "agents": [],
    }


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path


def write_result(results_dir, filename, result):
    (results_dir / filename).write_text(json.dumps(result))


def make_db(exists=False, experiment_id="abcd-1234"):
    db = AsyncMock()
    db.run_exists = AsyncMock(return_value=exists)
    db.save_run = AsyncMock(return_value=experiment_id)
    return db


class TestSync:
    @pytest.mark.asyncio
    async def test_uploads_new_run(self, results_dir):
        write_result(results_dir, "run1.json", make_result())
        db = make_db(exists=False)
        with patch("experiments.sync.SupabaseClient.connect", return_value=AsyncMock(return_value=db)) as mock_connect:
            mock_connect.return_value = db
            with patch("experiments.sync.SupabaseClient.connect", new=AsyncMock(return_value=db)):
                await sync(results_dir)
        db.save_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_existing_run(self, results_dir):
        write_result(results_dir, "run1.json", make_result())
        db = make_db(exists=True)
        with patch("experiments.sync.SupabaseClient.connect", new=AsyncMock(return_value=db)):
            await sync(results_dir)
        db.save_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_upload(self, results_dir):
        write_result(results_dir, "run1.json", make_result())
        db = make_db(exists=False)
        with patch("experiments.sync.SupabaseClient.connect", new=AsyncMock(return_value=db)):
            await sync(results_dir, dry_run=True)
        db.save_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_uploads_multiple_files(self, results_dir):
        for i in range(3):
            write_result(results_dir, f"run{i}.json", make_result(run_number=i, started_at=f"2026-05-20T10:0{i}:00+00:00"))
        db = make_db(exists=False)
        with patch("experiments.sync.SupabaseClient.connect", new=AsyncMock(return_value=db)):
            await sync(results_dir)
        assert db.save_run.call_count == 3

    @pytest.mark.asyncio
    async def test_empty_dir_prints_no_files(self, results_dir, capsys):
        db = make_db()
        with patch("experiments.sync.SupabaseClient.connect", new=AsyncMock(return_value=db)):
            await sync(results_dir)
        assert "No JSON files found" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_failed_upload_continues_to_next(self, results_dir):
        write_result(results_dir, "run1.json", make_result(run_number=1, started_at="2026-05-20T10:00:00+00:00"))
        write_result(results_dir, "run2.json", make_result(run_number=2, started_at="2026-05-20T10:01:00+00:00"))
        db = make_db(exists=False)
        db.save_run = AsyncMock(side_effect=[Exception("network error"), "ok-id"])
        with patch("experiments.sync.SupabaseClient.connect", new=AsyncMock(return_value=db)):
            await sync(results_dir)
        assert db.save_run.call_count == 2
