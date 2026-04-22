"""DGX Spark compute model — GPU/CPU execution latency estimation."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import numpy as np

from dgx_gp_spark_sim.config import DGXSparkConfig


@dataclass
class ComputeStats:
    """Running statistics for the compute model."""

    tasks_executed: int = 0
    total_compute_ms: float = 0.0
    batches_executed: int = 0
    active_batches: int = 0
    peak_active_batches: int = 0
    total_flops: float = 0.0


class DGXSparkComputeModel:
    """Simulates DGX Spark GPU/CPU compute behaviour.

    Given a task's FLOPs requirement, estimates execution latency using
    the configured peak throughput and a batch-efficiency curve that
    models diminishing returns at higher batch sizes.
    """

    def __init__(self, config: DGXSparkConfig | None = None) -> None:
        self._cfg = config or DGXSparkConfig()
        self._peak_tflops = self._cfg.gpu_tflops
        self._max_batches = self._cfg.max_concurrent_batches
        self._overhead_ms = self._cfg.compute_overhead_ms
        self._stats = ComputeStats()
        self._lock = asyncio.Lock()

        self._batch_sizes: list[float] = []
        self._batch_efficiencies: list[float] = []
        for point in self._cfg.batch_curve_points:
            self._batch_sizes.append(point[0])
            self._batch_efficiencies.append(point[1])

    @property
    def stats(self) -> ComputeStats:
        return self._stats

    @property
    def utilization(self) -> float:
        """Current compute utilization as fraction of max concurrent batches."""
        if self._max_batches == 0:
            return 0.0
        return self._stats.active_batches / self._max_batches

    def _batch_efficiency(self, batch_size: int) -> float:
        """Interpolate batch efficiency from the configured curve points."""
        if batch_size <= 0:
            return 1.0
        return float(np.interp(batch_size, self._batch_sizes, self._batch_efficiencies))

    def estimate_compute_ms(self, flops: float, batch_size: int = 1) -> float:
        """Estimate compute latency in milliseconds.

        Parameters
        ----------
        flops : float
            Total floating-point operations for the task.
        batch_size : int
            Current batch size (affects GPU efficiency).

        Returns
        -------
        float
            Estimated compute time in milliseconds.
        """
        peak_flops_per_ms = (self._peak_tflops * 1e12) / 1000.0
        if peak_flops_per_ms <= 0:
            return 0.0

        efficiency = self._batch_efficiency(batch_size)
        raw_ms = flops / (peak_flops_per_ms * efficiency)
        return raw_ms + self._overhead_ms

    def estimate_memory_pressure(self, input_bytes: int, batch_size: int = 1) -> float:
        """Estimate memory pressure as a fraction of unified memory.

        This is a simplified model: total working set = input_bytes * batch_size
        plus a baseline overhead, divided by total memory.
        """
        baseline_gb = 2.0
        working_set_gb = (input_bytes * batch_size) / (1024**3) + baseline_gb
        return min(1.0, working_set_gb / self._cfg.unified_memory_gb)

    async def execute(self, flops: float, input_bytes: int = 0, batch_size: int = 1) -> float:
        """Simulate task execution and return compute latency in ms.

        Respects max concurrent batch limits; callers may be delayed if
        all batch slots are occupied.
        """
        while self._stats.active_batches >= self._max_batches:
            await asyncio.sleep(0.001)

        async with self._lock:
            self._stats.active_batches += 1
            self._stats.peak_active_batches = max(
                self._stats.peak_active_batches, self._stats.active_batches
            )

        latency_ms = self.estimate_compute_ms(flops, batch_size)
        jitter = random.gauss(0, latency_ms * 0.02)
        latency_ms = max(0.001, latency_ms + jitter)

        await asyncio.sleep(latency_ms / 1000.0)

        async with self._lock:
            self._stats.active_batches -= 1
            self._stats.tasks_executed += 1
            self._stats.total_compute_ms += latency_ms
            self._stats.total_flops += flops
            if batch_size > 1:
                self._stats.batches_executed += 1

        return latency_ms

    def reset_stats(self) -> None:
        self._stats = ComputeStats()
