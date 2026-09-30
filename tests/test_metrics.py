"""Tests for latency percentile collection."""

from __future__ import annotations

from app.metrics import LatencyRecorder


def test_empty_recorder_reports_zeros() -> None:
    snapshot = LatencyRecorder().snapshot()
    assert snapshot == {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}


def test_percentiles_are_meaningful() -> None:
    recorder = LatencyRecorder()
    for value in range(1, 101):  # 1..100 ms
        recorder.record(value)

    snapshot = recorder.snapshot()
    assert snapshot["count"] == 100
    assert snapshot["p50_ms"] == 50
    assert snapshot["p95_ms"] == 95
    assert snapshot["max_ms"] == 100.0
    assert 50 <= snapshot["avg_ms"] <= 51


def test_ring_buffer_bounds_memory() -> None:
    recorder = LatencyRecorder(max_samples=10)
    for value in range(1000):
        recorder.record(value)
    assert recorder.count() == 10
    # Only the last ten samples survive, so the maximum is 999.
    assert recorder.snapshot()["max_ms"] == 999.0
