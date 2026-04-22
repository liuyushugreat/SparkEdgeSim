"""EdgeUnitNode — unified façade for a single DGX Spark + GP Spark edge unit."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from dgx_gp_spark_sim.config import EdgeUnitConfig
from dgx_gp_spark_sim.edge_unit.compute import DGXSparkComputeModel
from dgx_gp_spark_sim.edge_unit.storage import GPSparkStorageModel
from dgx_gp_spark_sim.models import (
    BatchRequest,
    BatchResult,
    HealthStatus,
    Task,
    TaskResult,
    TaskStatus,
)
from dgx_gp_spark_sim.network.model import NetworkModel
from dgx_gp_spark_sim.scheduler.scheduler import Scheduler
from dgx_gp_spark_sim.state.store import StateStore
from dgx_gp_spark_sim.telemetry.collector import MetricsCollector


class EdgeUnitNode:
    """A single edge compute unit that composes DGX Spark compute,
    GP Spark storage, network, scheduler, state, and telemetry.

    This is the primary entry point for task submission.  It receives
    tasks/batches, schedules them, simulates compute and state access,
    and returns detailed latency breakdowns.
    """

    def __init__(self, config: EdgeUnitConfig | None = None) -> None:
        self._cfg = config or EdgeUnitConfig()
        self._start_time = time.time()
        self._failed = False
        self._recovery_until: float = 0.0

        self.compute = DGXSparkComputeModel(self._cfg.dgx_spark)
        self.storage = GPSparkStorageModel(self._cfg.gp_spark)
        self.network = NetworkModel(self._cfg.network)
        self.scheduler = Scheduler(
            policy=self._cfg.scheduler_policy,
            queue_capacity=self._cfg.queue_capacity,
            microbatch_max=self._cfg.microbatch_max,
            microbatch_timeout_ms=self._cfg.microbatch_timeout_ms,
            staleness_bound_ms=self._cfg.staleness_bound_ms,
        )
        self.state = StateStore(
            gp_config=self._cfg.gp_spark,
            local_cache_capacity=self._cfg.local_cache_capacity,
            warm_state_capacity=self._cfg.warm_state_capacity,
            staleness_bound_ms=self._cfg.staleness_bound_ms,
        )
        self.metrics = MetricsCollector(unit_id=self._cfg.unit_id)

    @property
    def unit_id(self) -> str:
        return self._cfg.unit_id

    @property
    def config(self) -> EdgeUnitConfig:
        return self._cfg

    def _maybe_fail(self) -> bool:
        """Roll for stochastic failure based on configured failure_rate."""
        if self._cfg.failure_rate <= 0:
            return False
        return random.random() < self._cfg.failure_rate

    async def _wait_for_recovery(self) -> float:
        """If the unit is in a failed state, wait for recovery."""
        now = time.time()
        if now < self._recovery_until:
            wait_s = self._recovery_until - now
            await asyncio.sleep(wait_s)
            return wait_s * 1000.0
        return 0.0

    async def submit_task(self, task: Task) -> TaskResult:
        """Submit a single task for execution.

        Full pipeline: enqueue -> schedule -> state access -> compute -> result.
        """
        t0 = time.monotonic()

        recovery_ms = await self._wait_for_recovery()
        if recovery_ms > 0:
            self.metrics.record_recovery(recovery_ms)

        if self._maybe_fail():
            self._failed = True
            self._recovery_until = time.time() + self._cfg.recovery_time_ms / 1000.0
            self.metrics.record_task_failed()
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error="simulated_failure",
            )

        admitted = await self.scheduler.enqueue(task)
        if not admitted:
            self.metrics.record_task_dropped()
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.DROPPED,
                error="queue_full",
            )

        scheduled_task = await self.scheduler.dequeue()
        if scheduled_task is None:
            scheduled_task = task

        queue_delay_ms = (time.monotonic() - t0) * 1000.0

        state_access_ms = 0.0
        total_state_hit = 0
        total_state_read = 0
        for ref in task.state_refs:
            _val, lat, tier = await self.state.get(ref)
            state_access_ms += lat
            total_state_read += 1
            if tier in ("hot", "warm"):
                total_state_hit += 1
                self.metrics.record_state_access(hit=True)
            else:
                self.metrics.record_state_access(hit=False)

        compute_ms = await self.compute.execute(
            flops=task.flops,
            input_bytes=task.input_bytes,
            batch_size=1,
        )

        transfer_ms = 0.0
        if task.output_bytes > 0:
            transfer_ms = self.network.estimate_transfer_ms(task.output_bytes)

        total_ms = queue_delay_ms + state_access_ms + compute_ms + transfer_ms

        hit_ratio = total_state_hit / total_state_read if total_state_read > 0 else 1.0
        memory_pressure = self.compute.estimate_memory_pressure(task.input_bytes)

        self.metrics.record_task_complete(
            total_ms=total_ms,
            compute_ms=compute_ms,
            state_ms=state_access_ms,
            queue_ms=queue_delay_ms,
            transfer_ms=transfer_ms,
        )
        self.metrics.record_network_bytes(task.output_bytes)
        self.metrics.update_gauges(
            queue_length=self.scheduler.queue_length,
            compute_util=self.compute.utilization,
            storage_util=self.storage.utilization,
        )

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            latency_ms=round(total_ms, 3),
            compute_ms=round(compute_ms, 3),
            queue_delay_ms=round(queue_delay_ms, 3),
            state_access_ms=round(state_access_ms, 3),
            transfer_ms=round(transfer_ms, 3),
            state_hit_ratio=round(hit_ratio, 4),
            memory_pressure=round(memory_pressure, 4),
        )

    async def submit_batch(self, batch: BatchRequest) -> BatchResult:
        """Submit a batch of tasks.

        Enqueues all tasks, then flushes them as a micro-batch for
        amortized compute execution.
        """
        t0 = time.monotonic()
        results: list[TaskResult] = []
        batch_size = len(batch.tasks)

        for task in batch.tasks:
            await self.scheduler.enqueue(task)

        tasks = await self.scheduler.flush_batch()
        self.metrics.record_batch(len(tasks))

        for task in tasks:
            state_access_ms = 0.0
            total_state_hit = 0
            total_state_read = 0
            for ref in task.state_refs:
                _val, lat, tier = await self.state.get(ref)
                state_access_ms += lat
                total_state_read += 1
                hit = tier in ("hot", "warm")
                if hit:
                    total_state_hit += 1
                self.metrics.record_state_access(hit=hit)

            compute_ms = await self.compute.execute(
                flops=task.flops,
                input_bytes=task.input_bytes,
                batch_size=batch_size,
            )

            transfer_ms = self.network.estimate_transfer_ms(task.output_bytes)
            queue_ms = (time.monotonic() - t0) * 1000.0
            total_ms = queue_ms + state_access_ms + compute_ms + transfer_ms
            hit_ratio = total_state_hit / total_state_read if total_state_read > 0 else 1.0

            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                latency_ms=round(total_ms, 3),
                compute_ms=round(compute_ms, 3),
                queue_delay_ms=round(queue_ms, 3),
                state_access_ms=round(state_access_ms, 3),
                transfer_ms=round(transfer_ms, 3),
                state_hit_ratio=round(hit_ratio, 4),
            )
            results.append(result)

            self.metrics.record_task_complete(
                total_ms=total_ms,
                compute_ms=compute_ms,
                state_ms=state_access_ms,
                queue_ms=queue_ms,
                transfer_ms=transfer_ms,
            )
            self.metrics.record_network_bytes(task.output_bytes)

        total_latency = (time.monotonic() - t0) * 1000.0
        completed = sum(1 for r in results if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == TaskStatus.FAILED)

        self.metrics.update_gauges(
            queue_length=self.scheduler.queue_length,
            compute_util=self.compute.utilization,
            storage_util=self.storage.utilization,
        )

        return BatchResult(
            batch_id=batch.batch_id,
            results=results,
            total_latency_ms=round(total_latency, 3),
            tasks_completed=completed,
            tasks_failed=failed,
        )

    async def fetch_state(self, keys: list[str]) -> dict[str, Any]:
        """Fetch state values for the given keys."""
        result: dict[str, Any] = {}
        total_latency = 0.0
        for key in keys:
            val, lat, tier = await self.state.get(key)
            result[key] = {"value": val, "latency_ms": lat, "tier": tier}
            total_latency += lat
        return {"values": result, "total_latency_ms": total_latency}

    async def persist_audit(self, key: str, value: Any) -> float:
        """Persist a value and record to audit log. Returns write latency ms."""
        return await self.state.put(key, value)

    def get_metrics(self) -> dict[str, Any]:
        """Export current metrics snapshot."""
        return self.metrics.export()

    def health(self) -> HealthStatus:
        """Return current health status."""
        return HealthStatus(
            unit_id=self._cfg.unit_id,
            status="recovering" if time.time() < self._recovery_until else "healthy",
            uptime_s=round(time.time() - self._start_time, 2),
            queue_length=self.scheduler.queue_length,
            compute_utilization=round(self.compute.utilization, 4),
            storage_utilization=round(self.storage.utilization, 4),
        )

    def get_profile(self) -> dict[str, Any]:
        """Return the current hardware and system configuration."""
        return self._cfg.model_dump()

    def reconfigure(
        self,
        scheduler_policy: str | None = None,
        queue_capacity: int | None = None,
        microbatch_max: int | None = None,
        microbatch_timeout_ms: float | None = None,
        failure_rate: float | None = None,
    ) -> None:
        """Dynamically reconfigure runtime parameters."""
        if scheduler_policy is not None:
            self._cfg.scheduler_policy = scheduler_policy
        if queue_capacity is not None:
            self._cfg.queue_capacity = queue_capacity
            self.scheduler._capacity = queue_capacity
        if microbatch_max is not None:
            self._cfg.microbatch_max = microbatch_max
            self.scheduler._microbatch_max = microbatch_max
        if microbatch_timeout_ms is not None:
            self._cfg.microbatch_timeout_ms = microbatch_timeout_ms
            self.scheduler._timeout_ms = microbatch_timeout_ms
        if failure_rate is not None:
            self._cfg.failure_rate = failure_rate
