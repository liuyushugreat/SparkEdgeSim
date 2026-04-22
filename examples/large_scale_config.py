"""Example: 300 km² / 10K drones scenario configuration.

Demonstrates how to configure SparkEdgeSim for a large-scale UAM deployment
with multiple edge units across a wide area.
"""

import asyncio
import json
import math

from dgx_gp_spark_sim.config import EdgeUnitConfig, DGXSparkConfig, GPSparkConfig, NetworkConfig
from dgx_gp_spark_sim.edge_unit.node import EdgeUnitNode
from dgx_gp_spark_sim.models import Task
from dgx_gp_spark_sim.workloads.plugins import UAMNeighborQueryWorkload


# Scenario parameters
AREA_KM2 = 300
DRONES = 10_000
CELL_SIZE_KM = 1.0
NUM_CELLS = int(AREA_KM2 / (CELL_SIZE_KM ** 2))
DRONES_PER_CELL = DRONES / NUM_CELLS

# One edge unit per ~25 km² zone
EDGE_UNITS = int(math.ceil(AREA_KM2 / 25.0))


async def main() -> None:
    print(f"Scenario: {AREA_KM2} km² area, {DRONES:,} drones")
    print(f"  Cells:       {NUM_CELLS}")
    print(f"  Drones/cell: {DRONES_PER_CELL:.1f}")
    print(f"  Edge units:  {EDGE_UNITS}")
    print()

    # Create edge units with zone-specific configurations
    edges: list[EdgeUnitNode] = []
    for i in range(EDGE_UNITS):
        cfg = EdgeUnitConfig(
            unit_id=f"zone-{i:02d}",
            scheduler_policy="priority",
            queue_capacity=4096,
            microbatch_max=32,
            microbatch_timeout_ms=5.0,
            local_cache_capacity=50000,
            warm_state_capacity=500000,
            dgx_spark=DGXSparkConfig(
                gpu_tflops=1000.0,
                max_concurrent_batches=16,
            ),
            gp_spark=GPSparkConfig(
                iops=2_700_000,
                read_latency_us=20.0,
            ),
            network=NetworkConfig(
                edge_edge_rtt_ms=1.0,
                edge_cloud_rtt_ms=15.0,
                bandwidth_gbps=25.0,
            ),
        )
        edges.append(EdgeUnitNode(cfg))

    # Pre-populate state for each zone
    cells_per_zone = NUM_CELLS // EDGE_UNITS
    for zone_idx, edge in enumerate(edges):
        start_cell = zone_idx * cells_per_zone
        for c in range(cells_per_zone):
            cell_id = start_cell + c
            await edge.state.put(f"cell_{cell_id}", {
                "zone": zone_idx,
                "drone_count": int(DRONES_PER_CELL),
                "risk_level": 0.0,
            })

    # Generate and execute UAM workload on zone 0
    workload = UAMNeighborQueryWorkload()
    tasks = workload.generate(100)

    print(f"Executing {len(tasks)} UAM neighbor queries on zone-00...\n")

    results = []
    for task in tasks:
        result = await edges[0].submit_task(task)
        results.append(result)

    completed = sum(1 for r in results if r.status.value == "completed")
    latencies = [r.latency_ms for r in results]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    print(f"Results:")
    print(f"  Completed: {completed}/{len(tasks)}")
    print(f"  Avg latency: {avg_lat:.3f} ms")
    print(f"  Min latency: {min(latencies):.3f} ms")
    print(f"  Max latency: {max(latencies):.3f} ms")

    print(f"\nZone-00 metrics:")
    print(json.dumps(edges[0].get_metrics(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
