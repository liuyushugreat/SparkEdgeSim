"""
SparkEdgeSim — Interface-compatible, parameterized, hardware-aware
edge compute unit simulator for DGX Spark & GP Spark.

This package provides a discrete-event simulation of edge compute units
combining GPU/CPU compute (DGX Spark) with high-speed NVMe-oF storage
(GP Spark), exposing a stable API for integration with orchestration
frameworks such as SkyGrid or any custom edge computing platform.
"""

__version__ = "0.1.0"

from dgx_gp_spark_sim.models import (
    BatchRequest,
    BatchResult,
    Task,
    TaskResult,
)
from dgx_gp_spark_sim.client import EdgeUnitClient

__all__ = [
    "Task",
    "TaskResult",
    "BatchRequest",
    "BatchResult",
    "EdgeUnitClient",
]
