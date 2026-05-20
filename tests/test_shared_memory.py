import json
import pytest
import asyncio
from src.memory.shared import SharedMemoryStore


@pytest.mark.asyncio
async def test_record_and_retrieve():
    store = SharedMemoryStore()
    await store.record("A", 1, 2, 0)
    await store.record("A", 1, 3, 1)
    await store.record("B", 5, 5, 0)

    data = await store.get_all()
    assert data["A"] == [(1, 2, 0), (1, 3, 1)]
    assert data["B"] == [(5, 5, 0)]


@pytest.mark.asyncio
async def test_empty_store():
    store = SharedMemoryStore()
    data = await store.get_all()
    assert data == {}


@pytest.mark.asyncio
async def test_log_file_written_on_record(tmp_path):
    log = tmp_path / "sm.jsonl"
    store = SharedMemoryStore(log_path=log)
    await store.record("1", 2, 3, 0)
    await store.record("2", 5, 7, 1)

    lines = log.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"agent_id": "1", "x": 2, "y": 3, "t": 0}
    assert json.loads(lines[1]) == {"agent_id": "2", "x": 5, "y": 7, "t": 1}


@pytest.mark.asyncio
async def test_log_file_appends_incrementally(tmp_path):
    log = tmp_path / "sm.jsonl"
    store = SharedMemoryStore(log_path=log)
    await store.record("1", 1, 1, 0)
    assert log.read_text().count("\n") == 1
    await store.record("1", 1, 2, 1)
    assert log.read_text().count("\n") == 2


@pytest.mark.asyncio
async def test_no_log_file_when_path_is_none():
    store = SharedMemoryStore(log_path=None)
    await store.record("1", 1, 1, 0)  # should not raise


@pytest.mark.asyncio
async def test_concurrent_writes():
    store = SharedMemoryStore()

    async def write(agent_id, steps):
        for t, (x, y) in enumerate(steps):
            await store.record(agent_id, x, y, t)

    await asyncio.gather(
        write("A", [(0, 0), (0, 1), (0, 2)]),
        write("B", [(5, 5), (5, 6), (5, 7)]),
        write("C", [(9, 9), (9, 8)]),
    )

    data = await store.get_all()
    assert len(data["A"]) == 3
    assert len(data["B"]) == 3
    assert len(data["C"]) == 2