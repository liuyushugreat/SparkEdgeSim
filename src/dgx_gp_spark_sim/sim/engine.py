"""Discrete-event simulation engine for edge unit lifecycle management."""

from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine


class EventType(str, Enum):
    """Types of simulation events."""

    TASK_ARRIVE = "task_arrive"
    TASK_SCHEDULE = "task_schedule"
    TASK_COMPUTE_START = "task_compute_start"
    TASK_COMPUTE_DONE = "task_compute_done"
    STATE_READ = "state_read"
    STATE_WRITE = "state_write"
    NETWORK_SEND = "network_send"
    NETWORK_RECV = "network_recv"
    BATCH_FLUSH = "batch_flush"
    TIMEOUT = "timeout"
    FAILURE = "failure"
    RECOVERY = "recovery"
    CUSTOM = "custom"


@dataclass(order=True)
class SimEvent:
    """A single discrete event on the simulation timeline."""

    timestamp_ms: float
    sequence: int = field(compare=True)
    event_type: EventType = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)
    callback: Callable[..., Coroutine[Any, Any, None]] | None = field(
        default=None, compare=False
    )


class SimulationEngine:
    """A simple discrete-event simulation engine.

    Maintains an ordered event timeline and drives the simulation by
    processing events in chronological order. Supports both wall-clock
    mode (real asyncio.sleep) and fast-forward mode (virtual time only).

    Modes
    -----
    - **realtime**: Events trigger actual async sleeps proportional to
      their timestamps.  Useful when the edge unit runs as a live service.
    - **fast_forward**: Virtual clock advances instantly to each event's
      timestamp.  Useful for benchmarking and batch experiments.
    """

    def __init__(self, realtime: bool = True) -> None:
        self._realtime = realtime
        self._timeline: list[SimEvent] = []
        self._sequence = 0
        self._current_time_ms = 0.0
        self._running = False
        self._processed = 0
        self._event_log: list[dict[str, Any]] = []
        self._listeners: dict[EventType, list[Callable]] = {}

    @property
    def current_time_ms(self) -> float:
        return self._current_time_ms

    @property
    def events_processed(self) -> int:
        return self._processed

    @property
    def pending_events(self) -> int:
        return len(self._timeline)

    def schedule(
        self,
        event_type: EventType,
        delay_ms: float = 0.0,
        payload: dict[str, Any] | None = None,
        callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
    ) -> SimEvent:
        """Schedule an event at ``current_time + delay_ms``."""
        ts = self._current_time_ms + delay_ms
        event = SimEvent(
            timestamp_ms=ts,
            sequence=self._sequence,
            event_type=event_type,
            payload=payload or {},
            callback=callback,
        )
        heapq.heappush(self._timeline, event)
        self._sequence += 1
        return event

    def on(self, event_type: EventType, handler: Callable) -> None:
        """Register an event listener for a specific event type."""
        self._listeners.setdefault(event_type, []).append(handler)

    async def step(self) -> SimEvent | None:
        """Process the next event on the timeline."""
        if not self._timeline:
            return None

        event = heapq.heappop(self._timeline)

        if self._realtime and event.timestamp_ms > self._current_time_ms:
            wait_s = (event.timestamp_ms - self._current_time_ms) / 1000.0
            await asyncio.sleep(wait_s)

        self._current_time_ms = event.timestamp_ms

        if event.callback is not None:
            await event.callback(event)

        for handler in self._listeners.get(event.event_type, []):
            await handler(event)

        self._event_log.append({
            "time_ms": event.timestamp_ms,
            "type": event.event_type.value,
            "payload": event.payload,
        })
        self._processed += 1
        return event

    async def run(self, until_ms: float | None = None) -> int:
        """Run the simulation until the timeline is empty or until_ms is reached.

        Returns the number of events processed in this run.
        """
        self._running = True
        count = 0
        while self._running and self._timeline:
            if until_ms is not None and self._timeline[0].timestamp_ms > until_ms:
                break
            await self.step()
            count += 1
        self._running = False
        return count

    def stop(self) -> None:
        """Signal the engine to stop after processing the current event."""
        self._running = False

    def advance_time(self, delta_ms: float) -> None:
        """Manually advance virtual time (fast-forward mode only)."""
        self._current_time_ms += delta_ms

    def get_event_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent event log entries."""
        return self._event_log[-limit:]

    def export_trace(self) -> list[dict[str, Any]]:
        """Export the full event trace for analysis."""
        return list(self._event_log)

    def reset(self) -> None:
        """Clear timeline and reset clock."""
        self._timeline.clear()
        self._sequence = 0
        self._current_time_ms = 0.0
        self._processed = 0
        self._event_log.clear()
        self._running = False
