"""Metrics collector — aggregates and exports telemetry from all subsystems."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LatencyWindow:
    """Sliding window of latency samples for percentile computation."""

    _samples: list[float] = field(default_factory=list)
    _max_samples: int = 10000

    def record(self, value_ms: float) -> None:
        self._samples.append(value_ms)
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples:]

    def percentile(self, p: float) -> float:
        if not self._samples:
            return 0.0
        return float(np.percentile(self._samples, p))

    @property
    def count(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        self._samples.clear()


class MetricsCollector:
    """Central metrics collector that aggregates telemetry from edge unit subsystems.

    Tracks throughput, latency percentiles (p50/p95/p99), queue length,
    compute/storage utilization, network bytes, state hit/miss ratio,
    batch size distribution, recovery latency, and dropped/delayed task counts.
    """

    def __init__(self, unit_id: str = "edge-unit-0") -> None:
        self._unit_id = unit_id
        self._start_time = time.time()

        self.task_latency = LatencyWindow()
        self.compute_latency = LatencyWindow()
        self.state_latency = LatencyWindow()
        self.queue_delay = LatencyWindow()
        self.transfer_latency = LatencyWindow()
        self.batch_sizes = LatencyWindow()
        self.recovery_latency = LatencyWindow()

        self._tasks_completed = 0
        self._tasks_failed = 0
        self._tasks_dropped = 0
        self._tasks_delayed = 0
        self._bytes_transferred = 0

        self._state_hits = 0
        self._state_misses = 0

        self._current_queue_length = 0
        self._current_compute_util = 0.0
        self._current_storage_util = 0.0

    def record_task_complete(
        self,
        total_ms: float,
        compute_ms: float,
        state_ms: float,
        queue_ms: float,
        transfer_ms: float,
    ) -> None:
        """Record a completed task's latency breakdown."""
        self.task_latency.record(total_ms)
        self.compute_latency.record(compute_ms)
        self.state_latency.record(state_ms)
        self.queue_delay.record(queue_ms)
        self.transfer_latency.record(transfer_ms)
        self._tasks_completed += 1

    def record_task_failed(self) -> None:
        self._tasks_failed += 1

    def record_task_dropped(self) -> None:
        self._tasks_dropped += 1

    def record_task_delayed(self) -> None:
        self._tasks_delayed += 1

    def record_batch(self, size: int) -> None:
        self.batch_sizes.record(float(size))

    def record_state_access(self, hit: bool) -> None:
        if hit:
            self._state_hits += 1
        else:
            self._state_misses += 1

    def record_network_bytes(self, nbytes: int) -> None:
        self._bytes_transferred += nbytes

    def record_recovery(self, recovery_ms: float) -> None:
        self.recovery_latency.record(recovery_ms)

    def update_gauges(
        self,
        queue_length: int = 0,
        compute_util: float = 0.0,
        storage_util: float = 0.0,
    ) -> None:
        """Update current gauge values (called periodically)."""
        self._current_queue_length = queue_length
        self._current_compute_util = compute_util
        self._current_storage_util = storage_util

    @property
    def state_hit_ratio(self) -> float:
        total = self._state_hits + self._state_misses
        if total == 0:
            return 1.0
        return self._state_hits / total

    def throughput(self) -> float:
        """Tasks completed per second since startup."""
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._tasks_completed / elapsed

    def export(self) -> dict[str, Any]:
        """Export all metrics as a JSON-serializable dictionary."""
        return {
            "unit_id": self._unit_id,
            "uptime_s": time.time() - self._start_time,
            "throughput_tps": round(self.throughput(), 2),
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "tasks_dropped": self._tasks_dropped,
            "tasks_delayed": self._tasks_delayed,
            "latency": {
                "p50_ms": round(self.task_latency.percentile(50), 3),
                "p95_ms": round(self.task_latency.percentile(95), 3),
                "p99_ms": round(self.task_latency.percentile(99), 3),
                "compute_p50_ms": round(self.compute_latency.percentile(50), 3),
                "state_p50_ms": round(self.state_latency.percentile(50), 3),
                "queue_p50_ms": round(self.queue_delay.percentile(50), 3),
            },
            "queue_length": self._current_queue_length,
            "compute_utilization": round(self._current_compute_util, 4),
            "storage_utilization": round(self._current_storage_util, 4),
            "network_bytes_total": self._bytes_transferred,
            "state_hit_ratio": round(self.state_hit_ratio, 4),
            "batch_size_distribution": {
                "p50": round(self.batch_sizes.percentile(50), 1),
                "p95": round(self.batch_sizes.percentile(95), 1),
                "count": self.batch_sizes.count,
            },
            "recovery": {
                "count": self.recovery_latency.count,
                "p50_ms": round(self.recovery_latency.percentile(50), 3),
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._start_time = time.time()
        self.task_latency.reset()
        self.compute_latency.reset()
        self.state_latency.reset()
        self.queue_delay.reset()
        self.transfer_latency.reset()
        self.batch_sizes.reset()
        self.recovery_latency.reset()
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._tasks_dropped = 0
        self._tasks_delayed = 0
        self._bytes_transferred = 0
        self._state_hits = 0
        self._state_misses = 0
