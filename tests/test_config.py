"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest
import yaml

from dgx_gp_spark_sim.config import EdgeUnitConfig, load_config


def test_default_config() -> None:
    cfg = EdgeUnitConfig()
    assert cfg.unit_id == "edge-unit-0"
    assert cfg.dgx_spark.gpu_tflops == 1000.0
    assert cfg.gp_spark.iops == 2_700_000
    assert cfg.network.edge_edge_rtt_ms == 2.0


def test_load_config_from_yaml() -> None:
    data = {
        "unit_id": "test-unit",
        "scheduler_policy": "priority",
        "dgx_spark": {"gpu_tflops": 500.0},
        "network": {"bandwidth_gbps": 25.0},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name

    cfg = load_config(path)
    assert cfg.unit_id == "test-unit"
    assert cfg.scheduler_policy == "priority"
    assert cfg.dgx_spark.gpu_tflops == 500.0
    assert cfg.dgx_spark.cpu_cores == 20  # default preserved
    assert cfg.network.bandwidth_gbps == 25.0


def test_load_config_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_load_existing_profiles() -> None:
    configs_dir = Path(__file__).parent.parent / "configs"
    for yaml_file in configs_dir.glob("edge_unit_*.yaml"):
        cfg = load_config(yaml_file)
        assert cfg.unit_id != ""
        assert cfg.dgx_spark.gpu_tflops > 0
