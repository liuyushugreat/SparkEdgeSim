"""Tests for Scheduler."""

import pytest

from dgx_gp_spark_sim.models import Task
from dgx_gp_spark_sim.scheduler.scheduler import Scheduler


@pytest.fixture
def scheduler() -> Scheduler:
    return Scheduler(policy="fifo", queue_capacity=10, microbatch_max=4, microbatch_timeout_ms=50)


@pytest.mark.asyncio
async def test_enqueue_dequeue(scheduler: Scheduler) -> None:
    task = Task(task_id="t-1", flops=1e6)
    admitted = await scheduler.enqueue(task)
    assert admitted is True
    assert scheduler.queue_length == 1

    dequeued = await scheduler.dequeue()
    assert dequeued is not None
    assert dequeued.task_id == "t-1"
    assert scheduler.queue_length == 0


@pytest.mark.asyncio
async def test_queue_full_drops_task(scheduler: Scheduler) -> None:
    for i in range(10):
        await scheduler.enqueue(Task(task_id=f"t-{i}", flops=1e6))

    admitted = await scheduler.enqueue(Task(task_id="t-overflow", flops=1e6))
    assert admitted is False
    assert scheduler.tasks_dropped == 1


@pytest.mark.asyncio
async def test_priority_ordering() -> None:
    sched = Scheduler(policy="priority", queue_capacity=100)
    await sched.enqueue(Task(task_id="low", flops=1e6, priority=1))
    await sched.enqueue(Task(task_id="high", flops=1e6, priority=10))

    first = await sched.dequeue()
    assert first is not None
    assert first.task_id == "high"


@pytest.mark.asyncio
async def test_flush_batch(scheduler: Scheduler) -> None:
    for i in range(6):
        await scheduler.enqueue(Task(task_id=f"t-{i}", flops=1e6))

    batch = await scheduler.flush_batch()
    assert len(batch) == 4  # microbatch_max = 4


@pytest.mark.asyncio
async def test_flush_timeout() -> None:
    sched = Scheduler(policy="fifo", microbatch_max=8, microbatch_timeout_ms=20)
    await sched.enqueue(Task(task_id="t-0", flops=1e6))

    batch = await sched.flush_batch()
    assert len(batch) >= 1


def test_get_stats(scheduler: Scheduler) -> None:
    stats = scheduler.get_stats()
    assert stats["policy"] == "fifo"
    assert stats["queue_capacity"] == 10
