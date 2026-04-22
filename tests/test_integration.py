"""Integration tests — end-to-end task execution through EdgeUnitNode."""

import pytest

from dgx_gp_spark_sim.config import EdgeUnitConfig
from dgx_gp_spark_sim.edge_unit.node import EdgeUnitNode
from dgx_gp_spark_sim.models import BatchRequest, Task, TaskStatus


@pytest.fixture
def node() -> EdgeUnitNode:
    cfg = EdgeUnitConfig(
        unit_id="integration-test",
        failure_rate=0.0,
    )
    return EdgeUnitNode(cfg)


@pytest.fixture
def failing_node() -> EdgeUnitNode:
    cfg = EdgeUnitConfig(
        unit_id="fail-test",
        failure_rate=1.0,  # always fail
        recovery_time_ms=10.0,
    )
    return EdgeUnitNode(cfg)


@pytest.mark.asyncio
async def test_single_task_execution(node: EdgeUnitNode) -> None:
    await node.state.put("cell_12", {"risk": 0.3})

    task = Task(
        task_id="int-001",
        op_type="risk_score",
        flops=1e8,
        input_bytes=4096,
        state_refs=["cell_12"],
    )
    result = await node.submit_task(task)

    assert result.status == TaskStatus.COMPLETED
    assert result.latency_ms > 0
    assert result.compute_ms > 0
    assert result.state_hit_ratio == 1.0


@pytest.mark.asyncio
async def test_batch_execution(node: EdgeUnitNode) -> None:
    tasks = [
        Task(task_id=f"batch-{i}", flops=5e7, input_bytes=2048)
        for i in range(8)
    ]
    batch = BatchRequest(batch_id="int-batch-001", tasks=tasks)
    result = await node.submit_batch(batch)

    assert result.tasks_completed == 8
    assert result.total_latency_ms > 0


@pytest.mark.asyncio
async def test_state_miss_penalty(node: EdgeUnitNode) -> None:
    task = Task(
        task_id="miss-001",
        flops=1e6,
        state_refs=["nonexistent_state"],
    )
    result = await node.submit_task(task)
    assert result.state_hit_ratio == 0.0
    assert result.state_access_ms > 0


@pytest.mark.asyncio
async def test_failure_injection(failing_node: EdgeUnitNode) -> None:
    task = Task(task_id="fail-001", flops=1e6)
    result = await failing_node.submit_task(task)
    assert result.status == TaskStatus.FAILED
    assert result.error == "simulated_failure"


@pytest.mark.asyncio
async def test_metrics_after_execution(node: EdgeUnitNode) -> None:
    task = Task(task_id="metrics-001", flops=1e7)
    await node.submit_task(task)

    metrics = node.get_metrics()
    assert metrics["tasks_completed"] >= 1
    assert metrics["latency"]["p50_ms"] > 0


@pytest.mark.asyncio
async def test_health_check(node: EdgeUnitNode) -> None:
    health = node.health()
    assert health.status == "healthy"
    assert health.unit_id == "integration-test"


@pytest.mark.asyncio
async def test_reconfigure(node: EdgeUnitNode) -> None:
    node.reconfigure(scheduler_policy="priority", queue_capacity=2048)
    assert node.config.scheduler_policy == "priority"
    assert node.config.queue_capacity == 2048
