"""Latency measurement for the service.

Why percentiles and not the average: one slow request out of a hundred barely
moves the mean, yet it is exactly what users notice. ``p95`` ("95% of requests
were faster than this") describes the tail honestly.

The collector is a fixed-size ring buffer: memory use stays constant no matter
how long the process runs, which matters because the API is long-lived.
"""

from __future__ import annotations

from collections import deque


class LatencyRecorder:
    """Keeps the most recent latency samples and reports percentiles."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._samples: deque[float] = deque(maxlen=max(1, max_samples))

    def record(self, milliseconds: float) -> None:
        self._samples.append(float(milliseconds))

    def count(self) -> int:
        return len(self._samples)

    def percentile(self, fraction: float) -> float:
        """Nearest-rank percentile (no interpolation), 0.0 when empty."""
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples)
        index = max(0, min(len(ordered) - 1, int(round(fraction * len(ordered))) - 1))
        return round(ordered[index], 1)

    def snapshot(self) -> dict[str, float | int]:
        if not self._samples:
            return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        return {
            "count": len(self._samples),
            "avg_ms": round(sum(self._samples) / len(self._samples), 1),
            "p50_ms": self.percentile(0.50),
            "p95_ms": self.percentile(0.95),
            "max_ms": round(max(self._samples), 1),
        }
