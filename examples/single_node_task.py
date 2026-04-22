"""Example: Single node — submit individual tasks and inspect latency breakdown."""

import asyncio

from dgx_gp_spark_sim.edge_unit.node import EdgeUnitNode
from dgx_gp_spark_sim.models import Task


async def main() -> None:
    node = EdgeUnitNode()

    # Pre-populate some state
    await node.state.put("cell_12", {"risk_score": 0.42, "drone_count": 3})
    await node.state.put("neighbor_window_12", {"positions": [[1, 2], [3, 4], [5, 6]]})

    task = Task(
        task_id="t-001",
        op_type="risk_score",
        flops=1.2e8,
        input_bytes=4096,
        state_refs=["cell_12", "neighbor_window_12"],
        priority=1,
    )

    result = await node.submit_task(task)

    print(f"Task {result.task_id}")
    print(f"  Status:          {result.status}")
    print(f"  Total latency:   {result.latency_ms:.3f} ms")
    print(f"  Compute:         {result.compute_ms:.3f} ms")
    print(f"  Queue delay:     {result.queue_delay_ms:.3f} ms")
    print(f"  State access:    {result.state_access_ms:.3f} ms")
    print(f"  Transfer:        {result.transfer_ms:.3f} ms")
    print(f"  State hit ratio: {result.state_hit_ratio:.4f}")
    print(f"  Memory pressure: {result.memory_pressure:.4f}")

    print("\nMetrics:")
    import json
    print(json.dumps(node.get_metrics(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
