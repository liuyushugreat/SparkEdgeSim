"""Tests for DGXSparkComputeModel."""

import asyncio

import pytest

from dgx_gp_spark_sim.config import DGXSparkConfig
from dgx_gp_spark_sim.edge_unit.compute import DGXSparkComputeModel


@pytest.fixture
def compute() -> DGXSparkComputeModel:
    return DGXSparkComputeModel()


@pytest.fixture
def custom_compute() -> DGXSparkComputeModel:
    cfg = DGXSparkConfig(gpu_tflops=500.0, max_concurrent_batches=4)
    return DGXSparkComputeModel(cfg)


def test_estimate_compute_ms_positive(compute: DGXSparkComputeModel) -> None:
    latency = compute.estimate_compute_ms(flops=1e9, batch_size=1)
    assert latency > 0


def test_batch_efficiency_increases_latency(compute: DGXSparkComputeModel) -> None:
    lat_1 = compute.estimate_compute_ms(flops=1e9, batch_size=1)
    lat_32 = compute.estimate_compute_ms(flops=1e9, batch_size=32)
    assert lat_32 > lat_1


def test_memory_pressure_bounded(compute: DGXSparkComputeModel) -> None:
    pressure = compute.estimate_memory_pressure(input_bytes=1024, batch_size=1)
    assert 0 <= pressure <= 1.0


@pytest.mark.asyncio
async def test_execute_returns_latency(compute: DGXSparkComputeModel) -> None:
    latency = await compute.execute(flops=1e6, input_bytes=512)
    assert latency > 0
    assert compute.stats.tasks_executed == 1


@pytest.mark.asyncio
async def test_concurrent_batch_limit(custom_compute: DGXSparkComputeModel) -> None:
    tasks = [custom_compute.execute(flops=1e8) for _ in range(8)]
    results = await asyncio.gather(*tasks)
    assert all(r > 0 for r in results)
    assert custom_compute.stats.tasks_executed == 8


def test_reset_stats(compute: DGXSparkComputeModel) -> None:
    compute.reset_stats()
    assert compute.stats.tasks_executed == 0
    assert compute.stats.total_compute_ms == 0.0
