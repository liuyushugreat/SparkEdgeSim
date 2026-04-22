"""Core data models for tasks, batches, and results."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle status of a task."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DROPPED = "dropped"


class Task(BaseModel):
    """A single compute task submitted to an edge unit."""

    task_id: str
    op_type: str = "generic"
    flops: float = Field(default=1e6, description="Estimated floating-point operations")
    input_bytes: int = Field(default=1024, description="Input data size in bytes")
    output_bytes: int = Field(default=256, description="Output data size in bytes")
    state_refs: list[str] = Field(default_factory=list, description="State keys to access")
    priority: int = Field(default=0, description="Higher value = higher priority")
    deadline_ms: float | None = Field(default=None, description="Optional deadline in ms")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class TaskResult(BaseModel):
    """Result of executing a single task."""

    task_id: str
    status: TaskStatus = TaskStatus.COMPLETED
    latency_ms: float = 0.0
    compute_ms: float = 0.0
    queue_delay_ms: float = 0.0
    state_access_ms: float = 0.0
    transfer_ms: float = 0.0
    state_hit_ratio: float = 1.0
    memory_pressure: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchRequest(BaseModel):
    """A micro-batch of tasks submitted together."""

    batch_id: str
    tasks: list[Task]
    flush_policy: str = Field(default="immediate", description="immediate | timeout | size")
    max_wait_ms: float = Field(default=0.0, description="Max wait before flush (timeout policy)")


class BatchResult(BaseModel):
    """Aggregated result for a batch of tasks."""

    batch_id: str
    results: list[TaskResult]
    total_latency_ms: float = 0.0
    batch_overhead_ms: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0


class StateRequest(BaseModel):
    """Request to fetch state from an edge unit."""

    keys: list[str]
    requester_id: str = ""
    include_metadata: bool = False


class StateResponse(BaseModel):
    """Response carrying state data."""

    values: dict[str, Any] = Field(default_factory=dict)
    access_latency_ms: float = 0.0
    hit_ratio: float = 1.0
    source_tier: str = "hot"


class AuditEntry(BaseModel):
    """An audit log entry for state persistence."""

    entry_id: str
    timestamp: float = Field(default_factory=time.time)
    action: str
    key: str
    value_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """Health check response."""

    unit_id: str
    status: str = "healthy"
    uptime_s: float = 0.0
    queue_length: int = 0
    compute_utilization: float = 0.0
    storage_utilization: float = 0.0


class ReconfigureRequest(BaseModel):
    """Dynamic reconfiguration request."""

    scheduler_policy: str | None = None
    queue_capacity: int | None = None
    microbatch_max: int | None = None
    microbatch_timeout_ms: float | None = None
    failure_rate: float | None = None
