"""Example: Multi-edge cluster with a cloud control node."""

import asyncio
import json

from dgx_gp_spark_sim.config import EdgeUnitConfig, NetworkConfig
from dgx_gp_spark_sim.edge_unit.node import EdgeUnitNode
from dgx_gp_spark_sim.models import Task
from dgx_gp_spark_sim.network.model import LinkType


async def main() -> None:
    # Create 4 edge units and 1 cloud coordinator
    edge_configs = [
        EdgeUnitConfig(
            unit_id=f"edge-{i}",
            network=NetworkConfig(
                edge_edge_rtt_ms=1.0 + i * 0.5,
                edge_cloud_rtt_ms=15.0 + i * 2.0,
            ),
        )
        for i in range(4)
    ]

    edges = [EdgeUnitNode(cfg) for cfg in edge_configs]

    # Register links
    for i, edge_a in enumerate(edges):
        for j, edge_b in enumerate(edges):
            if i != j:
                edge_a.network.register_link(f"edge-{j}", LinkType.EDGE_EDGE)
        edge_a.network.register_link("cloud", LinkType.EDGE_CLOUD)

    # Pre-populate spatial state
    for i, edge in enumerate(edges):
        for cell in range(i * 25, (i + 1) * 25):
            await edge.state.put(f"cell_{cell}", {"zone": i, "load": 0.0})

    # Cloud dispatches tasks to each edge
    print("Cloud dispatching tasks to edge units...\n")
    all_results = []

    for i, edge in enumerate(edges):
        tasks = [
            Task(
                task_id=f"cloud-{i}-{t}",
                op_type="inference",
                flops=1e8,
                input_bytes=4096,
                state_refs=[f"cell_{i * 25 + t % 25}"],
                priority=t % 3,
            )
            for t in range(10)
        ]

        for task in tasks:
            result = await edge.submit_task(task)
            all_results.append(result)

    completed = sum(1 for r in all_results if r.status.value == "completed")
    avg_latency = sum(r.latency_ms for r in all_results) / len(all_results)

    print(f"Total tasks: {len(all_results)}")
    print(f"Completed:   {completed}")
    print(f"Avg latency: {avg_latency:.3f} ms")

    # Print per-edge metrics
    for i, edge in enumerate(edges):
        metrics = edge.get_metrics()
        print(f"\n--- Edge {i} ---")
        print(f"  Throughput:    {metrics['throughput_tps']:.1f} tps")
        print(f"  p50 latency:   {metrics['latency']['p50_ms']:.3f} ms")
        print(f"  p99 latency:   {metrics['latency']['p99_ms']:.3f} ms")
        print(f"  State hit:     {metrics['state_hit_ratio']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
