"""Tests for StateStore."""

import pytest

from dgx_gp_spark_sim.state.store import StateStore


@pytest.fixture
def store() -> StateStore:
    return StateStore(local_cache_capacity=100, warm_state_capacity=1000)


@pytest.mark.asyncio
async def test_put_and_get_hot(store: StateStore) -> None:
    await store.put("key1", "value1")
    val, lat, tier = await store.get("key1")
    assert val == "value1"
    assert tier == "hot"
    assert lat < 1.0  # should be very fast


@pytest.mark.asyncio
async def test_miss_returns_none(store: StateStore) -> None:
    val, lat, tier = await store.get("nonexistent")
    assert val is None
    assert tier == "miss"


@pytest.mark.asyncio
async def test_cold_tier_access(store: StateStore) -> None:
    store.preload_cold({"cold_key": "cold_value"})
    val, lat, tier = await store.get("cold_key")
    assert val == "cold_value"
    assert tier == "cold"

    # After access, should be promoted to hot
    val2, lat2, tier2 = await store.get("cold_key")
    assert tier2 == "hot"


@pytest.mark.asyncio
async def test_eviction_on_capacity(store: StateStore) -> None:
    for i in range(150):
        await store.put(f"key_{i}", f"val_{i}")
    assert store.stats.evictions > 0


@pytest.mark.asyncio
async def test_neighbor_state_lookup(store: StateStore) -> None:
    await store.put("cell_0", {"pos": [0, 0]})
    await store.put("cell_1", {"pos": [1, 0]})

    results = await store.get_neighbor_state(["cell_0", "cell_1", "cell_2"])
    assert "cell_0" in results
    assert "cell_1" in results
    assert results["cell_0"][2] == "hot"
    assert results["cell_2"][2] == "miss"


@pytest.mark.asyncio
async def test_snapshot_and_replay() -> None:
    store = StateStore(local_cache_capacity=100, warm_state_capacity=1000)
    await store.put("snap_key", "snap_val")

    snap = store.snapshot()
    assert "snap_key" in snap["hot"]

    # Clear only the data tiers, keep snapshots intact
    store._hot.clear()
    store._warm.clear()
    val, _, tier = await store.get("snap_key")
    assert val is None

    store.replay_snapshot(0)
    val2, _, tier2 = await store.get("snap_key")
    assert val2 == "snap_val"


def test_audit_log(store: StateStore) -> None:
    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_until_complete(store.put("audit_key", "audit_val"))
    loop.run_until_complete(store.get("audit_key"))

    log = store.get_audit_log(10)
    assert len(log) >= 2
    actions = [e["action"] for e in log]
    assert "write" in actions
    assert "read" in actions
    loop.close()


@pytest.mark.asyncio
async def test_hit_ratio(store: StateStore) -> None:
    await store.put("hit_key", "val")
    await store.get("hit_key")  # hit
    await store.get("miss_key")  # miss
    assert 0 < store.hit_ratio < 1.0
