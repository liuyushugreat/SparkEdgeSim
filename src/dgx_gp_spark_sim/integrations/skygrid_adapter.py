"""SkyGrid integration adapter.

Provides three operating modes for SkyGrid-style orchestration systems:

- **Mode 1 — Live Edge Unit**: SkyGrid submits operators/batches and
  queries state as if talking to a real edge unit.
- **Mode 2 — Performance Oracle**: Given task characteristics and data
  placement, returns predicted latency breakdowns without executing.
- **Mode 3 — Discrete-Event Backend**: SkyGrid manages global time
  advancement; this adapter only simulates internal unit behaviour
  within a single time step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dgx_gp_spark_sim.config import EdgeUnitConfig
from dgx_gp_spark_sim.edge_unit.node import EdgeUnitNode
from dgx_gp_spark_sim.models import BatchRequest, BatchResult, Task, TaskResult
from dgx_gp_spark_sim.network.model import LinkType


@dataclass
class LatencyBreakdown:
    """Predicted latency breakdown returned by the oracle mode."""

    compute_ms: float
    state_access_ms: float
    queue_ms: float
    transfer_ms: float
    total_ms: float


class SkyGridAdapter:
    """Adapter that bridges SkyGrid orchestration with SparkEdgeSim.

    Parameters
    ----------
    config : EdgeUnitConfig, optional
        Configuration for the underlying edge unit.
    mode : str
        Operating mode: ``"live"``, ``"oracle"``, or ``"discrete"``.
    """

    def __init__(
        self,
        config: EdgeUnitConfig | None = None,
        mode: str = "live",
    ) -> None:
        self._node = EdgeUnitNode(config)
        self._mode = mode
        self._virtual_time_ms = 0.0

    @property
    def node(self) -> EdgeUnitNode:
        """Access the underlying EdgeUnitNode directly."""
        return self._node

    # ── Mode 1: Live Edge Unit ──

    async def submit_operator_batch(self, batch: BatchRequest) -> BatchResult:
        """SkyGrid submits an operator batch for execution (Mode 1)."""
        return await self._node.submit_batch(batch)

    async def query_local_state(self, keys: list[str]) -> dict[str, Any]:
        """SkyGrid queries local state from this edge unit (Mode 1)."""
        return await self._node.fetch_state(keys)

    async def request_transfer(
        self, data_bytes: int, target_id: str, link_type: str = "edge_edge"
    ) -> float:
        """Simulate a data transfer to another node (Mode 1).

        Returns estimated transfer latency in ms.
        """
        lt = LinkType.EDGE_CLOUD if link_type == "edge_cloud" else LinkType.EDGE_EDGE
        return await self._node.network.transfer(data_bytes, remote_id=target_id, link_type=lt)

    def receive_metrics(self) -> dict[str, Any]:
        """SkyGrid pulls metrics from this edge unit (Mode 1)."""
        return self._node.get_metrics()

    # ── Mode 2: Performance Oracle ──

    def predict_latency(
        self,
        flops: float,
        input_bytes: int,
        output_bytes: int = 0,
        state_refs: list[str] | None = None,
        batch_size: int = 1,
        data_local: bool = True,
    ) -> LatencyBreakdown:
        """Predict latency breakdown without executing (Mode 2).

        Parameters
        ----------
        flops : float
            Estimated FLOPs for the task.
        input_bytes : int
            Input data size.
        output_bytes : int
            Output data size for transfer estimation.
        state_refs : list[str], optional
            State keys to access.
        batch_size : int
            Batch size for compute efficiency.
        data_local : bool
            If True, assume hot-tier state access; otherwise cold.
        """
        compute_ms = self._node.compute.estimate_compute_ms(flops, batch_size)

        state_refs = state_refs or []
        if data_local:
            state_ms = len(state_refs) * self._node.state.HOT_ACCESS_MS
        else:
            state_ms = len(state_refs) * self._node.state.COLD_ACCESS_MS

        queue_ms = 0.0
        if self._node.scheduler.queue_length > 0:
            avg_task_ms = compute_ms
            queue_ms = self._node.scheduler.queue_length * avg_task_ms * 0.1

        transfer_ms = 0.0
        if output_bytes > 0:
            transfer_ms = self._node.network.estimate_transfer_ms(output_bytes)

        total_ms = compute_ms + state_ms + queue_ms + transfer_ms

        return LatencyBreakdown(
            compute_ms=round(compute_ms, 3),
            state_access_ms=round(state_ms, 3),
            queue_ms=round(queue_ms, 3),
            transfer_ms=round(transfer_ms, 3),
            total_ms=round(total_ms, 3),
        )

    # ── Mode 3: Discrete-Event Backend ──

    async def step(self, task: Task, delta_ms: float = 0.0) -> TaskResult:
        """Process a single task within one discrete time step (Mode 3).

        SkyGrid manages the global clock; this method simulates
        internal unit behaviour for the given task.

        Parameters
        ----------
        task : Task
            The task to execute in this time step.
        delta_ms : float
            Virtual time to advance before processing.
        """
        self._virtual_time_ms += delta_ms
        return await self._node.submit_task(task)

    @property
    def virtual_time_ms(self) -> float:
        return self._virtual_time_ms

    def advance_time(self, delta_ms: float) -> None:
        """Advance virtual time without processing any tasks (Mode 3)."""
        self._virtual_time_ms += delta_ms
