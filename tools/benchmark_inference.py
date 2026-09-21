"""Inference latency benchmark placeholder (report p50/p95/p99)."""

import statistics
import time


def summarize_latencies_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("No samples supplied")
    ordered = sorted(samples)
    percentile = lambda p: ordered[min(len(ordered) - 1, int(p * len(ordered)))]
    return {"p50": percentile(0.50), "p95": percentile(0.95), "p99": percentile(0.99), "mean": statistics.mean(samples)}
