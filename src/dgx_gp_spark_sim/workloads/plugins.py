"""Built-in workload plugins that generate task streams for simulation."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from dgx_gp_spark_sim.models import Task


class WorkloadPlugin(ABC):
    """Base class for workload generators.

    Subclasses implement ``generate()`` to produce a list of tasks
    representing a specific workload pattern.
    """

    @abstractmethod
    def generate(self, count: int, **kwargs: object) -> list[Task]:
        """Generate ``count`` tasks for this workload type."""
        ...


class HybridPipelineWorkload(WorkloadPlugin):
    """Mixed compute + state pipeline (e.g., SkyGrid operator chain).

    Alternates between compute-heavy and state-heavy tasks to model
    a realistic hybrid edge pipeline.
    """

    def generate(self, count: int, **kwargs: object) -> list[Task]:
        tasks: list[Task] = []
        for i in range(count):
            if i % 2 == 0:
                tasks.append(Task(
                    task_id=f"hybrid-compute-{i}",
                    op_type="nn_inference",
                    flops=random.uniform(5e7, 5e8),
                    input_bytes=random.randint(2048, 16384),
                    output_bytes=random.randint(256, 2048),
                    state_refs=[],
                    priority=random.randint(0, 3),
                ))
            else:
                cell = random.randint(0, 99)
                tasks.append(Task(
                    task_id=f"hybrid-state-{i}",
                    op_type="state_update",
                    flops=random.uniform(1e6, 1e7),
                    input_bytes=random.randint(512, 4096),
                    output_bytes=256,
                    state_refs=[f"cell_{cell}", f"neighbor_window_{cell}"],
                    priority=random.randint(0, 2),
                ))
        return tasks


class UAMNeighborQueryWorkload(WorkloadPlugin):
    """UAM spatial neighbor query workload.

    Simulates queries for drone positions, risk scores, and trajectory
    data within spatial cells — typical of urban air mobility systems.
    """

    def generate(self, count: int, **kwargs: object) -> list[Task]:
        tasks: list[Task] = []
        for i in range(count):
            cell = random.randint(0, 299)
            neighbors = [f"cell_{cell + d}" for d in [-1, 0, 1] if 0 <= cell + d < 300]
            tasks.append(Task(
                task_id=f"uam-query-{i}",
                op_type="neighbor_query",
                flops=random.uniform(1e6, 5e7),
                input_bytes=random.randint(256, 2048),
                output_bytes=random.randint(1024, 8192),
                state_refs=neighbors,
                priority=1,
            ))
        return tasks


class StateHeavySymbolicWorkload(WorkloadPlugin):
    """State-heavy symbolic reasoning (rule evaluation, constraint checking).

    Each task accesses many state keys with minimal compute requirements.
    """

    def generate(self, count: int, **kwargs: object) -> list[Task]:
        tasks: list[Task] = []
        for i in range(count):
            num_refs = random.randint(5, 20)
            refs = [f"rule_{random.randint(0, 500)}" for _ in range(num_refs)]
            tasks.append(Task(
                task_id=f"symbolic-{i}",
                op_type="rule_eval",
                flops=random.uniform(1e5, 5e6),
                input_bytes=random.randint(128, 1024),
                output_bytes=128,
                state_refs=refs,
                priority=0,
            ))
        return tasks


class ComputeHeavyNNWorkload(WorkloadPlugin):
    """Compute-heavy neural network inference workload.

    Large FLOPs, large input tensors, minimal state access.
    """

    def generate(self, count: int, **kwargs: object) -> list[Task]:
        tasks: list[Task] = []
        for i in range(count):
            tasks.append(Task(
                task_id=f"nn-{i}",
                op_type="deep_inference",
                flops=random.uniform(1e8, 2e9),
                input_bytes=random.randint(8192, 65536),
                output_bytes=random.randint(512, 4096),
                state_refs=[],
                priority=random.randint(0, 5),
            ))
        return tasks
