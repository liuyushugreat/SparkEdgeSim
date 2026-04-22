"""Python SDK client for the SparkEdgeSim REST API."""

from __future__ import annotations

from typing import Any

import httpx

from dgx_gp_spark_sim.models import (
    BatchRequest,
    BatchResult,
    HealthStatus,
    StateRequest,
    StateResponse,
    Task,
    TaskResult,
)


class EdgeUnitClient:
    """Synchronous client for interacting with a running SparkEdgeSim server.

    Example
    -------
    >>> from dgx_gp_spark_sim import EdgeUnitClient, Task
    >>> client = EdgeUnitClient("http://localhost:8080")
    >>> task = Task(task_id="t-001", op_type="risk_score", flops=1.2e8)
    >>> result = client.submit_task(task)
    >>> print(result.latency_ms, result.compute_ms)
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)

    def submit_task(self, task: Task) -> TaskResult:
        """Submit a single task and return the result."""
        resp = self._client.post("/submit_task", json=task.model_dump())
        resp.raise_for_status()
        return TaskResult(**resp.json())

    def submit_batch(self, batch: BatchRequest) -> BatchResult:
        """Submit a batch of tasks and return aggregated results."""
        resp = self._client.post("/submit_batch", json=batch.model_dump())
        resp.raise_for_status()
        return BatchResult(**resp.json())

    def fetch_state(self, keys: list[str], requester_id: str = "") -> StateResponse:
        """Fetch state values for the given keys."""
        req = StateRequest(keys=keys, requester_id=requester_id)
        resp = self._client.post("/fetch_state", json=req.model_dump())
        resp.raise_for_status()
        return StateResponse(**resp.json())

    def persist_audit(self, key: str, value_hash: str, action: str = "write") -> dict[str, Any]:
        """Persist an audit entry."""
        from dgx_gp_spark_sim.models import AuditEntry

        entry = AuditEntry(entry_id=f"audit-{key}", action=action, key=key, value_hash=value_hash)
        resp = self._client.post("/persist_audit", json=entry.model_dump())
        resp.raise_for_status()
        return resp.json()

    def get_metrics(self) -> dict[str, Any]:
        """Fetch current metrics from the edge unit."""
        resp = self._client.get("/metrics")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> HealthStatus:
        """Check health status."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return HealthStatus(**resp.json())

    def get_profile(self) -> dict[str, Any]:
        """Get the hardware/system profile."""
        resp = self._client.get("/profile")
        resp.raise_for_status()
        return resp.json()

    def reconfigure(self, **kwargs: Any) -> dict[str, str]:
        """Dynamically reconfigure the edge unit."""
        resp = self._client.post("/control/reconfigure", json=kwargs)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> EdgeUnitClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncEdgeUnitClient:
    """Async client for interacting with a running SparkEdgeSim server."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def submit_task(self, task: Task) -> TaskResult:
        resp = await self._client.post("/submit_task", json=task.model_dump())
        resp.raise_for_status()
        return TaskResult(**resp.json())

    async def submit_batch(self, batch: BatchRequest) -> BatchResult:
        resp = await self._client.post("/submit_batch", json=batch.model_dump())
        resp.raise_for_status()
        return BatchResult(**resp.json())

    async def fetch_state(self, keys: list[str]) -> StateResponse:
        req = StateRequest(keys=keys)
        resp = await self._client.post("/fetch_state", json=req.model_dump())
        resp.raise_for_status()
        return StateResponse(**resp.json())

    async def get_metrics(self) -> dict[str, Any]:
        resp = await self._client.get("/metrics")
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> HealthStatus:
        resp = await self._client.get("/health")
        resp.raise_for_status()
        return HealthStatus(**resp.json())

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncEdgeUnitClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
