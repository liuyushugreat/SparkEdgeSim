"""Example: Two edge units communicating via simulated network."""

import asyncio

from dgx_gp_spark_sim.config import EdgeUnitConfig, NetworkConfig
from dgx_gp_spark_sim.edge_unit.node import EdgeUnitNode
from dgx_gp_spark_sim.models import Task
from dgx_gp_spark_sim.network.model import LinkType


async def main() -> None:
    cfg_a = EdgeUnitConfig(unit_id="edge-A")
    cfg_b = EdgeUnitConfig(unit_id="edge-B")

    node_a = EdgeUnitNode(cfg_a)
    node_b = EdgeUnitNode(cfg_b)

    # Register network links between the two units
    node_a.network.register_link("edge-B", LinkType.EDGE_EDGE)
    node_b.network.register_link("edge-A", LinkType.EDGE_EDGE)

    # Pre-populate state on node B
    await node_b.state.put("cell_50", {"risk": 0.8, "velocity": [10, 20]})

    # Node A needs state from node B — simulate remote fetch
    print("Node A requests state from Node B (via network)...")
    transfer_ms = await node_a.network.transfer(
        message_bytes=256,
        remote_id="edge-B",
        link_type=LinkType.EDGE_EDGE,
    )
    print(f"  Network transfer: {transfer_ms:.3f} ms")

    # Simulate node B reading its local state
    val, state_lat, tier = await node_b.state.get("cell_50")
    print(f"  State read on B:  {state_lat:.3f} ms (tier={tier})")

    # Transfer result back
    return_ms = await node_b.network.transfer(
        message_bytes=1024,
        remote_id="edge-A",
        link_type=LinkType.EDGE_EDGE,
    )
    print(f"  Return transfer:  {return_ms:.3f} ms")

    total_remote = transfer_ms + state_lat + return_ms
    print(f"  Total remote state access: {total_remote:.3f} ms")

    # Now node A executes a task using the fetched state
    task = Task(
        task_id="cross-001",
        op_type="risk_eval",
        flops=2e8,
        input_bytes=4096,
    )
    result = await node_a.submit_task(task)
    print(f"\nTask on Node A: {result.latency_ms:.3f} ms total")
    print(f"  Compute: {result.compute_ms:.3f} ms")

    # Summary
    print("\n--- Node A Metrics ---")
    import json
    print(json.dumps(node_a.get_metrics(), indent=2))

    print("\n--- Node B Metrics ---")
    print(json.dumps(node_b.get_metrics(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
