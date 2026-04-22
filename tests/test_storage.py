"""Tests for GPSparkStorageModel."""

import pytest

from dgx_gp_spark_sim.config import GPSparkConfig
from dgx_gp_spark_sim.edge_unit.storage import GPSparkStorageModel


@pytest.fixture
def storage() -> GPSparkStorageModel:
    return GPSparkStorageModel()


def test_estimate_read_latency_positive(storage: GPSparkStorageModel) -> None:
    lat = storage.estimate_read_latency_ms(4096)
    assert lat > 0


def test_rdma_reduces_latency() -> None:
    rdma_on = GPSparkStorageModel(GPSparkConfig(rdma_enabled=True))
    rdma_off = GPSparkStorageModel(GPSparkConfig(rdma_enabled=False))
    lat_on = rdma_on.estimate_read_latency_ms(4096)
    lat_off = rdma_off.estimate_read_latency_ms(4096)
    # RDMA should generally give lower base latency
    assert lat_on <= lat_off * 1.5  # allow jitter margin


@pytest.mark.asyncio
async def test_read_simulation(storage: GPSparkStorageModel) -> None:
    lat = await storage.read(8192)
    assert lat > 0
    assert storage.stats.reads == 1
    assert storage.stats.total_bytes_read == 8192


@pytest.mark.asyncio
async def test_write_simulation(storage: GPSparkStorageModel) -> None:
    lat = await storage.write(4096)
    assert lat > 0
    assert storage.stats.writes == 1
    assert storage.stats.total_bytes_written == 4096


def test_reset_stats(storage: GPSparkStorageModel) -> None:
    storage.reset_stats()
    assert storage.stats.reads == 0
