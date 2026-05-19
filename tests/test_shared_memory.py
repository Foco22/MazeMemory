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