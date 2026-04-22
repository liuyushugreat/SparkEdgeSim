"""Tests for MetricsCollector."""

from dgx_gp_spark_sim.telemetry.collector import MetricsCollector


def test_record_and_export() -> None:
    mc = MetricsCollector(unit_id="test-unit")

    mc.record_task_complete(
        total_ms=10.0,
        compute_ms=5.0,
        state_ms=2.0,
        queue_ms=1.0,
        transfer_ms=2.0,
    )
    mc.record_state_access(hit=True)
    mc.record_state_access(hit=False)
    mc.record_network_bytes(4096)
    mc.record_batch(8)

    metrics = mc.export()
    assert metrics["unit_id"] == "test-unit"
    assert metrics["tasks_completed"] == 1
    assert metrics["network_bytes_total"] == 4096
    assert 0.0 < metrics["state_hit_ratio"] < 1.0
    assert metrics["latency"]["p50_ms"] == 10.0


def test_throughput() -> None:
    mc = MetricsCollector()
    for _ in range(100):
        mc.record_task_complete(1.0, 0.5, 0.2, 0.1, 0.2)
    assert mc.throughput() > 0


def test_percentiles() -> None:
    mc = MetricsCollector()
    for i in range(100):
        mc.record_task_complete(float(i), float(i) * 0.5, 0.1, 0.05, 0.05)

    metrics = mc.export()
    assert metrics["latency"]["p50_ms"] < metrics["latency"]["p95_ms"]
    assert metrics["latency"]["p95_ms"] < metrics["latency"]["p99_ms"]


def test_reset() -> None:
    mc = MetricsCollector()
    mc.record_task_complete(5.0, 3.0, 1.0, 0.5, 0.5)
    mc.reset()
    assert mc.export()["tasks_completed"] == 0
