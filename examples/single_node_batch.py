"""Example: Single node — submit a micro-batch of tasks."""

import asyncio

from dgx_gp_spark_sim.edge_unit.node import EdgeUnitNode
from dgx_gp_spark_sim.models import BatchRequest, Task


async def main() -> None:
    node = EdgeUnitNode()

    tasks = [
        Task(
            task_id=f"batch-{i:03d}",
            op_type="classify",
            flops=5e7,
            input_bytes=2048,
            output_bytes=512,
        )
        for i in range(16)
    ]

    batch = BatchRequest(batch_id="demo-batch-001", tasks=tasks)
    result = await node.submit_batch(batch)

    print(f"Batch {result.batch_id}")
    print(f"  Tasks completed: {result.tasks_completed}")
    print(f"  Tasks failed:    {result.tasks_failed}")
    print(f"  Total latency:   {result.total_latency_ms:.3f} ms")
    print()

    for r in result.results:
        print(f"  {r.task_id}: {r.latency_ms:.3f} ms "
              f"(compute={r.compute_ms:.3f}, queue={r.queue_delay_ms:.3f})")

    print("\nMetrics:")
    import json
    print(json.dumps(node.get_metrics(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
