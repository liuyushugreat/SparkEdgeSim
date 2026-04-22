"""FastAPI REST server for the edge unit simulator."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException

from dgx_gp_spark_sim.config import EdgeUnitConfig, load_config
from dgx_gp_spark_sim.edge_unit.node import EdgeUnitNode
from dgx_gp_spark_sim.models import (
    AuditEntry,
    BatchRequest,
    BatchResult,
    HealthStatus,
    ReconfigureRequest,
    StateRequest,
    StateResponse,
    Task,
    TaskResult,
)

def create_app(config: EdgeUnitConfig | None = None) -> FastAPI:
    """Create and return a configured FastAPI application.

    Parameters
    ----------
    config : EdgeUnitConfig, optional
        Edge unit configuration. Uses defaults if not provided.
    """
    node = EdgeUnitNode(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.node = node
        yield

    app = FastAPI(
        title="SparkEdgeSim",
        description="DGX & GP Spark Edge Compute Unit Simulator API",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Also set eagerly so TestClient (which triggers lifespan) and
    # direct attribute access both work.
    app.state.node = node

    def _get_node() -> EdgeUnitNode:
        n: EdgeUnitNode | None = getattr(app.state, "node", None)
        if n is None:
            raise HTTPException(status_code=503, detail="Edge unit not initialized")
        return n

    @app.post("/submit_task", response_model=TaskResult)
    async def submit_task(task: Task) -> TaskResult:
        """Submit a single compute task to the edge unit."""
        return await _get_node().submit_task(task)

    @app.post("/submit_batch", response_model=BatchResult)
    async def submit_batch(batch: BatchRequest) -> BatchResult:
        """Submit a micro-batch of tasks."""
        return await _get_node().submit_batch(batch)

    @app.post("/fetch_state", response_model=StateResponse)
    async def fetch_state(req: StateRequest) -> StateResponse:
        """Fetch state values from the local state store."""
        n = _get_node()
        result = await n.fetch_state(req.keys)
        values = {k: v["value"] for k, v in result["values"].items()}
        total_lat = result["total_latency_ms"]
        hit_count = sum(
            1 for v in result["values"].values() if v["tier"] in ("hot", "warm")
        )
        total_keys = len(req.keys) or 1
        return StateResponse(
            values=values,
            access_latency_ms=total_lat,
            hit_ratio=hit_count / total_keys,
        )

    @app.post("/persist_audit")
    async def persist_audit(entry: AuditEntry) -> dict[str, Any]:
        """Persist a value to the state store with audit logging."""
        n = _get_node()
        latency = await n.persist_audit(entry.key, entry.value_hash)
        return {"status": "ok", "write_latency_ms": latency}

    @app.post("/control/reconfigure")
    async def reconfigure(req: ReconfigureRequest) -> dict[str, str]:
        """Dynamically reconfigure the edge unit."""
        n = _get_node()
        n.reconfigure(
            scheduler_policy=req.scheduler_policy,
            queue_capacity=req.queue_capacity,
            microbatch_max=req.microbatch_max,
            microbatch_timeout_ms=req.microbatch_timeout_ms,
            failure_rate=req.failure_rate,
        )
        return {"status": "reconfigured"}

    @app.get("/metrics")
    async def get_metrics() -> dict[str, Any]:
        """Export current telemetry metrics."""
        return _get_node().get_metrics()

    @app.get("/health", response_model=HealthStatus)
    async def health() -> HealthStatus:
        """Health check endpoint."""
        return _get_node().health()

    @app.get("/profile")
    async def profile() -> dict[str, Any]:
        """Return the current hardware and system profile."""
        return _get_node().get_profile()

    return app
