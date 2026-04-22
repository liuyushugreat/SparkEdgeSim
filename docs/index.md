# SparkEdgeSim Documentation

Welcome to the SparkEdgeSim documentation.

## What is SparkEdgeSim?

SparkEdgeSim is an **interface-compatible, parameterized, hardware-aware edge compute unit simulator** that models the combined capabilities of:

- **NVIDIA DGX Spark** — GPU/CPU compute with Grace Blackwell architecture
- **GP Spark** — NVMe-oF storage with RDMA hardware offload

## Getting Started

1. [Installation](../README.md#installation)
2. [Quick Start](../README.md#quick-start)
3. [Configuration Guide](../README.md#configuration)
4. [API Reference](../README.md#api-reference)
5. [SkyGrid Integration](../README.md#skygrid-integration)

## Design Principles

1. **Interface-compatible**: External callers interact with the simulator exactly as they would with real hardware.
2. **Hardware-calibrated**: Parameters are derived from published DGX Spark and GP Spark specifications.
3. **Event-driven**: Discrete-event simulation for accurate timing behavior.
4. **Pluggable**: Workloads, hardware profiles, scheduling policies, and storage models are all replaceable.
5. **Research-grade**: Designed as a reproducible artifact for academic publications.
