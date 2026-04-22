"""Network model — edge-edge, edge-cloud, and control-plane communication."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from enum import StrEnum

from dgx_gp_spark_sim.config import NetworkConfig


class LinkType(StrEnum):
    """Type of network link."""

    EDGE_EDGE = "edge_edge"
    EDGE_CLOUD = "edge_cloud"


@dataclass
class NetworkStats:
    """Running statistics for network transfers."""

    messages_sent: int = 0
    total_bytes: int = 0
    total_transfer_ms: float = 0.0
    congestion_events: int = 0


@dataclass
class NetworkLink:
    """A point-to-point network link with its own stats."""

    link_type: LinkType
    remote_id: str
    stats: NetworkStats = field(default_factory=NetworkStats)


class NetworkModel:
    """Simulates network communication latency and throughput.

    Transfer delay = RTT/2 + (message_size + overhead) / effective_bandwidth
    where effective_bandwidth accounts for congestion_factor and jitter.
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        self._cfg = config or NetworkConfig()
        self._links: dict[str, NetworkLink] = {}
        self._global_stats = NetworkStats()
        self._lock = asyncio.Lock()
        self._congestion_hooks: list = []

    @property
    def stats(self) -> NetworkStats:
        return self._global_stats

    def register_link(self, remote_id: str, link_type: LinkType) -> None:
        """Register a network link to a remote node."""
        self._links[remote_id] = NetworkLink(link_type=link_type, remote_id=remote_id)

    def add_congestion_hook(self, hook: object) -> None:
        """Register a callback invoked when congestion_factor > threshold."""
        self._congestion_hooks.append(hook)

    def _base_rtt_ms(self, link_type: LinkType) -> float:
        if link_type == LinkType.EDGE_CLOUD:
            return self._cfg.edge_cloud_rtt_ms
        return self._cfg.edge_edge_rtt_ms

    def estimate_transfer_ms(
        self,
        message_bytes: int,
        link_type: LinkType = LinkType.EDGE_EDGE,
    ) -> float:
        """Estimate one-way transfer latency for a message.

        Parameters
        ----------
        message_bytes : int
            Payload size in bytes (overhead added automatically).
        link_type : LinkType
            Whether this is edge-edge or edge-cloud.

        Returns
        -------
        float
            Estimated transfer time in milliseconds.
        """
        one_way_rtt_ms = self._base_rtt_ms(link_type) / 2.0
        jitter = random.gauss(0, self._cfg.jitter_ms)
        total_bytes = message_bytes + self._cfg.packet_overhead_bytes
        bw_bytes_per_ms = (self._cfg.bandwidth_gbps * 1e9 / 8) / 1000.0
        effective_bw = bw_bytes_per_ms / self._cfg.congestion_factor
        if effective_bw <= 0:
            effective_bw = 1.0
        transfer_ms = total_bytes / effective_bw
        return max(0.01, one_way_rtt_ms + transfer_ms + jitter)

    async def transfer(
        self,
        message_bytes: int,
        remote_id: str = "",
        link_type: LinkType = LinkType.EDGE_EDGE,
    ) -> float:
        """Simulate a network transfer and return latency in ms."""
        latency_ms = self.estimate_transfer_ms(message_bytes, link_type)
        await asyncio.sleep(latency_ms / 1000.0)

        async with self._lock:
            self._global_stats.messages_sent += 1
            self._global_stats.total_bytes += message_bytes
            self._global_stats.total_transfer_ms += latency_ms

            if remote_id and remote_id in self._links:
                link = self._links[remote_id]
                link.stats.messages_sent += 1
                link.stats.total_bytes += message_bytes
                link.stats.total_transfer_ms += latency_ms

        if self._cfg.congestion_factor > 1.5:
            async with self._lock:
                self._global_stats.congestion_events += 1

        return latency_ms

    def update_congestion(self, factor: float) -> None:
        """Dynamically update the congestion factor."""
        self._cfg.congestion_factor = max(1.0, factor)

    def reset_stats(self) -> None:
        self._global_stats = NetworkStats()
        for link in self._links.values():
            link.stats = NetworkStats()
