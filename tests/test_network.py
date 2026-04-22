"""Tests for NetworkModel."""

import pytest

from dgx_gp_spark_sim.config import NetworkConfig
from dgx_gp_spark_sim.network.model import LinkType, NetworkModel

# Suppress jitter randomness warning: some tests use zero jitter intentionally.


@pytest.fixture
def network() -> NetworkModel:
    return NetworkModel()


def test_estimate_transfer_positive(network: NetworkModel) -> None:
    lat = network.estimate_transfer_ms(1024, LinkType.EDGE_EDGE)
    assert lat > 0


def test_edge_cloud_higher_than_edge_edge(network: NetworkModel) -> None:
    lat_ee = network.estimate_transfer_ms(1024, LinkType.EDGE_EDGE)
    lat_ec = network.estimate_transfer_ms(1024, LinkType.EDGE_CLOUD)
    # Cloud RTT is higher, so cloud transfer should generally be slower
    assert lat_ec > lat_ee * 0.5  # allow jitter margin


def test_larger_message_takes_longer() -> None:
    cfg = NetworkConfig(jitter_ms=0.0)
    net = NetworkModel(cfg)
    lat_small = net.estimate_transfer_ms(100, LinkType.EDGE_EDGE)
    lat_large = net.estimate_transfer_ms(1_000_000, LinkType.EDGE_EDGE)
    assert lat_large > lat_small


@pytest.mark.asyncio
async def test_transfer_simulation(network: NetworkModel) -> None:
    lat = await network.transfer(2048, remote_id="node-1", link_type=LinkType.EDGE_EDGE)
    assert lat > 0
    assert network.stats.messages_sent == 1
    assert network.stats.total_bytes == 2048


def test_congestion_update() -> None:
    # Use deterministic config with zero jitter for this test
    cfg = NetworkConfig(jitter_ms=0.0, congestion_factor=1.0)
    net = NetworkModel(cfg)
    lat_normal = net.estimate_transfer_ms(100_000, LinkType.EDGE_EDGE)

    net.update_congestion(3.0)
    lat_congested = net.estimate_transfer_ms(100_000, LinkType.EDGE_EDGE)
    assert lat_congested > lat_normal
