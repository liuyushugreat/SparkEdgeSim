"""Configuration models for hardware profiles and edge unit parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DGXSparkConfig(BaseModel):
    """Hardware profile for NVIDIA DGX Spark compute unit."""

    gpu_tflops: float = Field(default=1000.0, description="Peak tensor TFLOPS (FP8/FP4)")
    cpu_cores: int = Field(default=20, description="Arm CPU core count")
    unified_memory_gb: float = Field(default=128.0, description="LPDDR5x unified memory in GB")
    memory_bandwidth_gbps: float = Field(default=273.0, description="Memory bandwidth in GB/s")
    nvme_capacity_tb: float = Field(default=4.0, description="Local NVMe storage in TB")
    nic_bandwidth_gbps: float = Field(default=200.0, description="ConnectX-7 NIC bandwidth in Gbps")
    max_concurrent_batches: int = Field(default=8, description="Max concurrent micro-batches on GPU")
    batch_curve_points: list[list[float]] = Field(
        default=[[1, 1.0], [4, 0.85], [8, 0.75], [16, 0.65], [32, 0.55]],
        description="[batch_size, efficiency] interpolation points",
    )
    power_watts: float = Field(default=240.0, description="TDP in watts")
    compute_overhead_ms: float = Field(default=0.05, description="Fixed kernel launch overhead")


class GPSparkConfig(BaseModel):
    """Hardware profile for GP Spark NVMe-oF storage unit."""

    max_ssd_count: int = Field(default=4, description="Number of M.2 SSD bays")
    ssd_capacity_tb: float = Field(default=8.0, description="Per-SSD capacity in TB")
    storage_bandwidth_gbps: float = Field(default=11.6, description="Aggregate throughput in GB/s")
    iops: int = Field(default=2_700_000, description="Peak IOPS")
    read_latency_us: float = Field(default=20.0, description="Read access latency in µs")
    write_latency_us: float = Field(default=25.0, description="Write access latency in µs")
    rdma_enabled: bool = Field(default=True, description="RDMA (RoCEv2) offload")
    gds_enabled: bool = Field(default=True, description="GPUDirect Storage support")
    nvmeof_enabled: bool = Field(default=True, description="NVMe-oF protocol enabled")
    power_watts: float = Field(default=100.0, description="TDP in watts")


class NetworkConfig(BaseModel):
    """Network parameters for edge communication."""

    edge_edge_rtt_ms: float = Field(default=2.0, description="Edge-to-edge RTT in ms")
    edge_cloud_rtt_ms: float = Field(default=20.0, description="Edge-to-cloud RTT in ms")
    bandwidth_gbps: float = Field(default=10.0, description="Link bandwidth in Gbps")
    jitter_ms: float = Field(default=0.5, description="Jitter standard deviation in ms")
    packet_overhead_bytes: int = Field(default=64, description="Per-packet protocol overhead")
    congestion_factor: float = Field(
        default=1.0, description="Multiplier for congestion (1.0 = no congestion)"
    )


class EdgeUnitConfig(BaseModel):
    """Configuration for a single edge compute unit."""

    unit_id: str = Field(default="edge-unit-0")
    scheduler_policy: str = Field(default="fifo", description="fifo | priority | edf | custom")
    queue_capacity: int = Field(default=1024, description="Max tasks in queue")
    staleness_bound_ms: float = Field(default=500.0, description="Max staleness for cached state")
    microbatch_max: int = Field(default=16, description="Max tasks per micro-batch")
    microbatch_timeout_ms: float = Field(default=10.0, description="Flush timeout for batching")
    local_cache_capacity: int = Field(default=10000, description="Hot state cache entries")
    warm_state_capacity: int = Field(default=100000, description="Warm state (GP Spark) entries")
    failure_rate: float = Field(default=0.0, description="Probability of failure per task [0,1]")
    recovery_time_ms: float = Field(default=500.0, description="Recovery time after failure")

    dgx_spark: DGXSparkConfig = Field(default_factory=DGXSparkConfig)
    gp_spark: GPSparkConfig = Field(default_factory=GPSparkConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)


def load_config(path: str | Path) -> EdgeUnitConfig:
    """Load edge unit configuration from a YAML file.

    Unspecified fields fall back to built-in defaults so partial YAML
    files are perfectly valid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return EdgeUnitConfig(**raw)
