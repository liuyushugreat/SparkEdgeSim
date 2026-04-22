"""Local task scheduler with pluggable policies and micro-batch assembly."""

from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from dgx_gp_spark_sim.models import Task


class SchedulerPolicy(StrEnum):
    """Built-in scheduling policies."""

    FIFO = "fifo"
    PRIORITY = "priority"
    EDF = "edf"  # Earliest Deadline First


@dataclass(order=True)
class _QueueEntry:
    """Internal priority-queue wrapper around a Task."""

    sort_key: float
    sequence: int
    task: Task = field(compare=False)


class Scheduler:
    """Single-node task scheduler with micro-batch support.

    Supports FIFO, priority-based, and EDF scheduling. Tasks can be
    dequeued individually or flushed as micro-batches when either
    ``microbatch_max`` tasks accumulate or ``microbatch_timeout_ms`` elapses.
    """

    def __init__(
        self,
        policy: str = "fifo",
        queue_capacity: int = 1024,
        microbatch_max: int = 16,
        microbatch_timeout_ms: float = 10.0,
        staleness_bound_ms: float = 500.0,
    ) -> None:
        self._policy = SchedulerPolicy(policy)
        self._capacity = queue_capacity
        self._microbatch_max = microbatch_max
        self._timeout_ms = microbatch_timeout_ms
        self._staleness_bound_ms = staleness_bound_ms

        self._heap: list[_QueueEntry] = []
        self._sequence = 0
        self._lock = asyncio.Lock()

        self._tasks_enqueued = 0
        self._tasks_dequeued = 0
        self._tasks_dropped = 0
        self._batches_flushed = 0

        self._backpressure = False
        self._admission_hook: Callable[[Task], bool] | None = None

    @property
    def queue_length(self) -> int:
        return len(self._heap)

    @property
    def is_full(self) -> bool:
        return len(self._heap) >= self._capacity

    @property
    def backpressure(self) -> bool:
        return self._backpressure

    @property
    def tasks_dropped(self) -> int:
        return self._tasks_dropped

    def set_admission_hook(self, hook: Callable[[Task], bool]) -> None:
        """Set a custom admission control function.

        The hook receives a Task and returns True to admit, False to reject.
        """
        self._admission_hook = hook

    def _sort_key(self, task: Task) -> float:
        if self._policy == SchedulerPolicy.PRIORITY:
            return -task.priority
        if self._policy == SchedulerPolicy.EDF:
            return task.deadline_ms if task.deadline_ms is not None else float("inf")
        return 0.0  # FIFO relies on sequence number

    async def enqueue(self, task: Task) -> bool:
        """Add a task to the scheduling queue.

        Returns True if admitted, False if dropped (queue full or rejected).
        """
        async with self._lock:
            if len(self._heap) >= self._capacity:
                self._tasks_dropped += 1
                self._backpressure = True
                return False

            if self._admission_hook and not self._admission_hook(task):
                self._tasks_dropped += 1
                return False

            entry = _QueueEntry(
                sort_key=self._sort_key(task),
                sequence=self._sequence,
                task=task,
            )
            heapq.heappush(self._heap, entry)
            self._sequence += 1
            self._tasks_enqueued += 1

            self._backpressure = len(self._heap) >= int(self._capacity * 0.9)

        return True

    async def dequeue(self) -> Task | None:
        """Remove and return the highest-priority task, or None if empty."""
        async with self._lock:
            if not self._heap:
                return None
            entry = heapq.heappop(self._heap)
            self._tasks_dequeued += 1
            self._backpressure = len(self._heap) >= int(self._capacity * 0.9)
            return entry.task

    async def flush_batch(self) -> list[Task]:
        """Collect up to ``microbatch_max`` tasks as a micro-batch.

        If fewer tasks are available, waits up to ``microbatch_timeout_ms``
        for more to arrive before returning whatever is ready.
        """
        batch: list[Task] = []
        deadline = time.monotonic() + self._timeout_ms / 1000.0

        while len(batch) < self._microbatch_max:
            task = await self.dequeue()
            if task is not None:
                batch.append(task)
            else:
                if time.monotonic() >= deadline:
                    break
                await asyncio.sleep(0.001)
                if time.monotonic() >= deadline:
                    break

        if batch:
            self._batches_flushed += 1

        return batch

    def get_stats(self) -> dict:
        """Return scheduler statistics as a dictionary."""
        return {
            "policy": self._policy.value,
            "queue_length": self.queue_length,
            "queue_capacity": self._capacity,
            "tasks_enqueued": self._tasks_enqueued,
            "tasks_dequeued": self._tasks_dequeued,
            "tasks_dropped": self._tasks_dropped,
            "batches_flushed": self._batches_flushed,
            "backpressure": self._backpressure,
        }
