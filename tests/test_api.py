"""Tests for the FastAPI REST server."""

import pytest
from fastapi.testclient import TestClient

from dgx_gp_spark_sim.api.server import create_app
from dgx_gp_spark_sim.config import EdgeUnitConfig


@pytest.fixture
def client() -> TestClient:
    app = create_app(EdgeUnitConfig(unit_id="test-api-unit"))
    return TestClient(app)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["unit_id"] == "test-api-unit"
    assert data["status"] == "healthy"


def test_profile(client: TestClient) -> None:
    resp = client.get("/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert "dgx_spark" in data
    assert "gp_spark" in data


def test_submit_task(client: TestClient) -> None:
    resp = client.post("/submit_task", json={
        "task_id": "api-task-1",
        "op_type": "test",
        "flops": 1e7,
        "input_bytes": 1024,
        "state_refs": [],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "api-task-1"
    assert data["status"] == "completed"
    assert data["latency_ms"] > 0


def test_submit_batch(client: TestClient) -> None:
    resp = client.post("/submit_batch", json={
        "batch_id": "api-batch-1",
        "tasks": [
            {"task_id": f"bt-{i}", "flops": 1e6, "input_bytes": 512}
            for i in range(4)
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["batch_id"] == "api-batch-1"
    assert data["tasks_completed"] == 4


def test_metrics(client: TestClient) -> None:
    client.post("/submit_task", json={"task_id": "m-1", "flops": 1e6})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tasks_completed"] >= 1


def test_fetch_state(client: TestClient) -> None:
    resp = client.post("/fetch_state", json={"keys": ["nonexistent_key"]})
    assert resp.status_code == 200


def test_reconfigure(client: TestClient) -> None:
    resp = client.post("/control/reconfigure", json={
        "scheduler_policy": "priority",
        "queue_capacity": 2048,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "reconfigured"


def test_persist_audit(client: TestClient) -> None:
    resp = client.post("/persist_audit", json={
        "entry_id": "a-1",
        "action": "write",
        "key": "test_key",
        "value_hash": "abc123",
    })
    assert resp.status_code == 200
