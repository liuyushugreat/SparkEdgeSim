"""GP Spark storage model — NVMe-oF / RDMA storage access simulation."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from dgx_gp_spark_sim.config import GPSparkConfig


@dataclass
class StorageStats:
    """Running statistics for the storage model."""

    reads: int = 0
    writes: int = 0
    total_read_latency_ms: float = 0.0
    total_write_latency_ms: float = 0.0
    total_bytes_read: int = 0
    total_bytes_written: int = 0


class GPSparkStorageModel:
    """Simulates GP Spark NVMe-oF storage access latency and throughput.

    Models three access paths:
    - RDMA direct (lowest latency when enabled)
    - GDS / GPUDirect Storage (GPU-to-storage, bypasses CPU)
    - Standard NVMe-oF
    """

    def __init__(self, config: GPSparkConfig | None = None) -> None:
        self._cfg = config or GPSparkConfig()
        self._stats = StorageStats()
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> StorageStats:
        return self._stats

    @property
    def utilization(self) -> float:
        """Rough utilization based on recent IOPS vs peak."""
        total_ops = self._stats.reads + self._stats.writes
        if total_ops == 0 or self._cfg.iops == 0:
            return 0.0
        total_time_s = max(
            (self._stats.total_read_latency_ms + self._stats.total_write_latency_ms) / 1000.0,
            0.001,
        )
        achieved_iops = total_ops / total_time_s
        return min(1.0, achieved_iops / self._cfg.iops)

    def estimate_read_latency_ms(self, size_bytes: int = 4096) -> float:
        """Estimate read latency for a given payload size.

        Combines base access latency with bandwidth-limited transfer time.
        """
        base_us = self._cfg.read_latency_us
        if self._cfg.rdma_enabled:
            base_us *= 0.8
        transfer_us = (size_bytes / (self._cfg.storage_bandwidth_gbps * 1e9 / 8)) * 1e6
        total_us = base_us + transfer_us
        jitter = random.gauss(0, total_us * 0.05)
        return max(0.001, (total_us + jitter) / 1000.0)

    def estimate_write_latency_ms(self, size_bytes: int = 4096) -> float:
        """Estimate write latency for a given payload size."""
        base_us = self._cfg.write_latency_us
        if self._cfg.rdma_enabled:
            base_us *= 0.85
        transfer_us = (size_bytes / (self._cfg.storage_bandwidth_gbps * 1e9 / 8)) * 1e6
        total_us = base_us + transfer_us
        jitter = random.gauss(0, total_us * 0.05)
        return max(0.001, (total_us + jitter) / 1000.0)

    async def read(self, size_bytes: int = 4096) -> float:
        """Simulate a storage read and return latency in ms."""
        latency_ms = self.estimate_read_latency_ms(size_bytes)
        await asyncio.sleep(latency_ms / 1000.0)
        async with self._lock:
            self._stats.reads += 1
            self._stats.total_read_latency_ms += latency_ms
            self._stats.total_bytes_read += size_bytes
        return latency_ms

    async def write(self, size_bytes: int = 4096) -> float:
        """Simulate a storage write and return latency in ms."""
        latency_ms = self.estimate_write_latency_ms(size_bytes)
        await asyncio.sleep(latency_ms / 1000.0)
        async with self._lock:
            self._stats.writes += 1
            self._stats.total_write_latency_ms += latency_ms
            self._stats.total_bytes_written += size_bytes
        return latency_ms

    def reset_stats(self) -> None:
        self._stats = StorageStats()
