"""Tiered state store — hot (memory) / warm (GP Spark) / cold (remote) access."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from dgx_gp_spark_sim.config import GPSparkConfig


@dataclass
class StateStats:
    """Running statistics for the state store."""

    hot_hits: int = 0
    warm_hits: int = 0
    cold_hits: int = 0
    total_reads: int = 0
    total_writes: int = 0
    total_read_latency_ms: float = 0.0
    total_write_latency_ms: float = 0.0
    evictions: int = 0
    audit_entries: int = 0
    snapshots_taken: int = 0


@dataclass
class AuditLogEntry:
    """Internal audit log entry."""

    timestamp: float
    action: str  # "read" | "write" | "evict" | "snapshot"
    key: str
    value_hash: str = ""
    tier: str = ""


class StateStore:
    """Tiered state storage with hot (LRU cache), warm (GP Spark), and cold (remote) tiers.

    - **Hot tier**: In-memory LRU cache, sub-microsecond access.
    - **Warm tier**: GP Spark NVMe-oF backed storage, ~20 µs access.
    - **Cold tier**: Remote fallback (simulated via configurable latency).

    Supports audit logging, snapshot/replay, and neighbour state lookups.
    """

    HOT_ACCESS_MS = 0.001  # ~1 µs
    COLD_ACCESS_MS = 5.0   # remote fallback

    def __init__(
        self,
        gp_config: GPSparkConfig | None = None,
        local_cache_capacity: int = 10000,
        warm_state_capacity: int = 100000,
        staleness_bound_ms: float = 500.0,
    ) -> None:
        self._gp_cfg = gp_config or GPSparkConfig()
        self._cache_cap = local_cache_capacity
        self._warm_cap = warm_state_capacity
        self._staleness_ms = staleness_bound_ms

        self._hot: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._warm: dict[str, tuple[Any, float]] = {}
        self._cold: dict[str, Any] = {}

        self._stats = StateStats()
        self._audit_log: list[AuditLogEntry] = []
        self._snapshots: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> StateStats:
        return self._stats

    @property
    def hit_ratio(self) -> float:
        """Hot + warm hit ratio over total reads."""
        if self._stats.total_reads == 0:
            return 1.0
        return (self._stats.hot_hits + self._stats.warm_hits) / self._stats.total_reads

    def _warm_read_ms(self) -> float:
        return self._gp_cfg.read_latency_us / 1000.0

    def _warm_write_ms(self) -> float:
        return self._gp_cfg.write_latency_us / 1000.0

    def _value_hash(self, value: Any) -> str:
        return hashlib.md5(str(value).encode()).hexdigest()[:8]

    def _append_audit(self, action: str, key: str, value: Any = None, tier: str = "") -> None:
        entry = AuditLogEntry(
            timestamp=time.time(),
            action=action,
            key=key,
            value_hash=self._value_hash(value) if value is not None else "",
            tier=tier,
        )
        self._audit_log.append(entry)
        self._stats.audit_entries += 1

    async def get(self, key: str) -> tuple[Any, float, str]:
        """Read a state value. Returns (value, access_latency_ms, tier).

        Checks hot -> warm -> cold. On warm/cold hit the value is
        promoted to the hot tier.
        """
        async with self._lock:
            self._stats.total_reads += 1

            # Hot tier
            if key in self._hot:
                val, ts = self._hot[key]
                self._hot.move_to_end(key)
                self._stats.hot_hits += 1
                latency = self.HOT_ACCESS_MS
                self._stats.total_read_latency_ms += latency
                self._append_audit("read", key, val, "hot")
                return val, latency, "hot"

            # Warm tier
            if key in self._warm:
                val, ts = self._warm[key]
                self._stats.warm_hits += 1
                latency = self._warm_read_ms()
                self._stats.total_read_latency_ms += latency
                self._promote_to_hot(key, val)
                self._append_audit("read", key, val, "warm")
                return val, latency, "warm"

            # Cold tier
            if key in self._cold:
                val = self._cold[key]
                self._stats.cold_hits += 1
                latency = self.COLD_ACCESS_MS
                self._stats.total_read_latency_ms += latency
                self._promote_to_hot(key, val)
                self._append_audit("read", key, val, "cold")
                return val, latency, "cold"

            latency = self.COLD_ACCESS_MS
            self._stats.cold_hits += 1
            self._stats.total_read_latency_ms += latency
            self._append_audit("read", key, tier="miss")
            return None, latency, "miss"

    async def put(self, key: str, value: Any) -> float:
        """Write a state value to the hot tier and persist to warm.

        Returns the write latency in ms (hot write + warm persist).
        """
        async with self._lock:
            self._stats.total_writes += 1
            now = time.time()

            self._promote_to_hot(key, value, timestamp=now)

            warm_latency = self._warm_write_ms()
            self._warm[key] = (value, now)
            if len(self._warm) > self._warm_cap:
                oldest_key = next(iter(self._warm))
                del self._warm[oldest_key]

            total_latency = self.HOT_ACCESS_MS + warm_latency
            self._stats.total_write_latency_ms += total_latency
            self._append_audit("write", key, value, "hot+warm")
            return total_latency

    def _promote_to_hot(
        self, key: str, value: Any, timestamp: float | None = None
    ) -> None:
        """Promote a key to the hot tier, evicting LRU if necessary."""
        ts = timestamp or time.time()
        self._hot[key] = (value, ts)
        self._hot.move_to_end(key)
        while len(self._hot) > self._cache_cap:
            evicted_key, _ = self._hot.popitem(last=False)
            self._stats.evictions += 1
            self._append_audit("evict", evicted_key, tier="hot")

    async def get_neighbor_state(self, keys: list[str]) -> dict[str, tuple[Any, float, str]]:
        """Batch-read multiple state keys (spatial neighbour lookup).

        Returns a dict mapping each key to (value, latency_ms, tier).
        """
        results: dict[str, tuple[Any, float, str]] = {}
        for key in keys:
            results[key] = await self.get(key)
        return results

    def snapshot(self) -> dict[str, Any]:
        """Take a point-in-time snapshot of the hot + warm tiers."""
        snap = {
            "timestamp": time.time(),
            "hot": {k: v for k, (v, _ts) in self._hot.items()},
            "warm": {k: v for k, (v, _ts) in self._warm.items()},
        }
        self._snapshots.append(snap)
        self._stats.snapshots_taken += 1
        self._append_audit("snapshot", key="*", tier="all")
        return snap

    def replay_snapshot(self, index: int = -1) -> None:
        """Restore state from a previously taken snapshot."""
        if not self._snapshots:
            return
        snap = self._snapshots[index]
        now = time.time()
        self._hot.clear()
        for k, v in snap.get("hot", {}).items():
            self._hot[k] = (v, now)
        self._warm.clear()
        for k, v in snap.get("warm", {}).items():
            self._warm[k] = (v, now)

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Return recent audit log entries as dicts."""
        entries = self._audit_log[-limit:]
        return [
            {
                "timestamp": e.timestamp,
                "action": e.action,
                "key": e.key,
                "value_hash": e.value_hash,
                "tier": e.tier,
            }
            for e in entries
        ]

    def preload_cold(self, data: dict[str, Any]) -> None:
        """Bulk-load data into the cold tier (for simulation setup)."""
        self._cold.update(data)

    def reset(self) -> None:
        """Clear all tiers and statistics."""
        self._hot.clear()
        self._warm.clear()
        self._cold.clear()
        self._stats = StateStats()
        self._audit_log.clear()
        self._snapshots.clear()
