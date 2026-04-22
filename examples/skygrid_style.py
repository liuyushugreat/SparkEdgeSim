"""Example: SkyGrid-style integration using the SkyGridAdapter.

Demonstrates all three operating modes:
- Mode 1: Live edge unit
- Mode 2: Performance oracle
- Mode 3: Discrete-event backend
"""

import asyncio
import json

from dgx_gp_spark_sim.config import EdgeUnitConfig
from dgx_gp_spark_sim.integrations.skygrid_adapter import SkyGridAdapter
from dgx_gp_spark_sim.models import BatchRequest, Task


async def main() -> None:
    adapter = SkyGridAdapter(
        config=EdgeUnitConfig(unit_id="skygrid-edge-0"),
        mode="live",
    )

    # Pre-populate state for the simulation
    await adapter.node.state.put("cell_12", {"risk": 0.3, "drones": 5})
    await adapter.node.state.put("cell_13", {"risk": 0.1, "drones": 2})

    # ── Mode 1: Live Edge Unit ──
    print("=" * 60)
    print("MODE 1: Live Edge Unit")
    print("=" * 60)

    batch = BatchRequest(
        batch_id="skygrid-batch-001",
        tasks=[
            Task(
                task_id=f"op-{i}",
                op_type="operator_eval",
                flops=8e7,
                input_bytes=2048,
                state_refs=["cell_12", "cell_13"],
            )
            for i in range(4)
        ],
    )

    result = await adapter.submit_operator_batch(batch)
    print(f"Batch completed: {result.tasks_completed} tasks, {result.total_latency_ms:.3f} ms")

    state = await adapter.query_local_state(["cell_12", "cell_13"])
    print(f"State query: {len(state['values'])} keys, {state['total_latency_ms']:.3f} ms")

    transfer_ms = await adapter.request_transfer(
        data_bytes=8192, target_id="edge-1", link_type="edge_edge"
    )
    print(f"Transfer to edge-1: {transfer_ms:.3f} ms")

    metrics = adapter.receive_metrics()
    print(f"Metrics: {metrics['tasks_completed']} tasks, p50={metrics['latency']['p50_ms']:.3f} ms")

    # ── Mode 2: Performance Oracle ──
    print("\n" + "=" * 60)
    print("MODE 2: Performance Oracle")
    print("=" * 60)

    breakdown = adapter.predict_latency(
        flops=2e8,
        input_bytes=4096,
        output_bytes=1024,
        state_refs=["cell_12", "cell_13"],
        batch_size=4,
        data_local=True,
    )
    print(f"Predicted breakdown:")
    print(f"  Compute:  {breakdown.compute_ms:.3f} ms")
    print(f"  State:    {breakdown.state_access_ms:.3f} ms")
    print(f"  Queue:    {breakdown.queue_ms:.3f} ms")
    print(f"  Transfer: {breakdown.transfer_ms:.3f} ms")
    print(f"  Total:    {breakdown.total_ms:.3f} ms")

    breakdown_remote = adapter.predict_latency(
        flops=2e8,
        input_bytes=4096,
        state_refs=["cell_12"],
        data_local=False,
    )
    print(f"\nWith remote state: {breakdown_remote.total_ms:.3f} ms "
          f"(vs local: {breakdown.total_ms:.3f} ms)")

    # ── Mode 3: Discrete-Event Backend ──
    print("\n" + "=" * 60)
    print("MODE 3: Discrete-Event Backend")
    print("=" * 60)

    for step in range(5):
        task = Task(
            task_id=f"discrete-{step}",
            op_type="step_eval",
            flops=5e7,
            input_bytes=1024,
            state_refs=["cell_12"],
        )
        result = await adapter.step(task, delta_ms=10.0)
        print(f"  Step {step}: t={adapter.virtual_time_ms:.1f} ms, "
              f"latency={result.latency_ms:.3f} ms")


if __name__ == "__main__":
    asyncio.run(main())
