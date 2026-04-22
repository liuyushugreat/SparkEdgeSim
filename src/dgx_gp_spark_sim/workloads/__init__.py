"""Workload plugins — predefined task generators for various scenarios."""

from dgx_gp_spark_sim.workloads.plugins import (
    ComputeHeavyNNWorkload,
    HybridPipelineWorkload,
    StateHeavySymbolicWorkload,
    UAMNeighborQueryWorkload,
    WorkloadPlugin,
)

__all__ = [
    "WorkloadPlugin",
    "HybridPipelineWorkload",
    "UAMNeighborQueryWorkload",
    "StateHeavySymbolicWorkload",
    "ComputeHeavyNNWorkload",
]
