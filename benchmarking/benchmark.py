"""
SPECTRA - Benchmarking Framework
MODULE K

Provides accurate, real-measurement benchmark infrastructure.
Never fabricates numbers — all results come from actual execution.
"""

from __future__ import annotations

import time
import statistics
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Any

from utils.logger import get_logger

log = get_logger("BENCHMARK")


@dataclass
class BenchmarkResult:
    """Results from a benchmark run. All values are real measurements."""
    model_id: str
    model_name: str
    runtime: str
    device: str
    provider: str

    warmup_iterations: int
    benchmark_iterations: int

    # All latency measurements in milliseconds
    latencies_ms: list[float]

    # Computed statistics
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float  # requests per second

    # Resource usage
    memory_before_mb: Optional[float]
    memory_after_mb: Optional[float]
    memory_delta_mb: Optional[float]

    # Metadata
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "runtime": self.runtime,
            "device": self.device,
            "provider": self.provider,
            "warmup_iterations": self.warmup_iterations,
            "benchmark_iterations": self.benchmark_iterations,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "throughput_rps": round(self.throughput_rps, 2),
            "memory_before_mb": self.memory_before_mb,
            "memory_after_mb": self.memory_after_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "timestamp": self.timestamp,
            "success": self.success,
            "error_message": self.error_message,
            "notes": self.notes,
        }

    def print_report(self) -> None:
        divider = "-" * 50
        print(f"\n{divider}")
        print(f"  BENCHMARK RESULT - {self.model_name}")
        print(f"{divider}")
        print(f"  Runtime   : {self.runtime}")
        print(f"  Device    : {self.device}")
        print(f"  Provider  : {self.provider}")
        print(f"  Timestamp : {self.timestamp}")
        if self.success:
            print(f"\n  Iterations: {self.benchmark_iterations} (warmup: {self.warmup_iterations})")
            print(f"\n  Latency")
            print(f"    Average : {self.avg_latency_ms:.2f} ms")
            print(f"    Min     : {self.min_latency_ms:.2f} ms")
            print(f"    Max     : {self.max_latency_ms:.2f} ms")
            print(f"    P50     : {self.p50_latency_ms:.2f} ms")
            print(f"    P95     : {self.p95_latency_ms:.2f} ms")
            print(f"    P99     : {self.p99_latency_ms:.2f} ms")
            print(f"\n  Throughput: {self.throughput_rps:.2f} req/s")
            if self.memory_delta_mb is not None:
                print(f"\n  Memory Delta: +{self.memory_delta_mb:.1f} MB")
        else:
            print(f"\n  STATUS: FAILED")
            print(f"  Error: {self.error_message}")
        print()


def _measure_memory_mb() -> Optional[float]:
    try:
        import psutil, os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _compute_percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = min(lower + 1, n - 1)
    fraction = index - lower
    return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])


def run_benchmark(
    inference_fn: Callable[[], Any],
    model_id: str,
    model_name: str,
    runtime: str,
    device: str,
    provider: str,
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
    notes: str = "",
) -> BenchmarkResult:
    """
    Run a benchmark on any inference function.

    Args:
        inference_fn: A callable that performs one inference pass.
        Other args: metadata for the result.

    Returns:
        BenchmarkResult with real measured latencies.
    """
    log.info(f"Starting benchmark: {model_name} | {device} | warmup={warmup_iterations} | iters={benchmark_iterations}")

    mem_before = _measure_memory_mb()
    latencies: list[float] = []
    error_msg: Optional[str] = None

    # Warmup
    log.info(f"Warmup phase: {warmup_iterations} iterations")
    for i in range(warmup_iterations):
        try:
            inference_fn()
        except Exception as e:
            log.error(f"Warmup iteration {i+1} failed: {e}")

    # Benchmark
    log.info(f"Benchmark phase: {benchmark_iterations} iterations")
    for i in range(benchmark_iterations):
        try:
            t0 = time.perf_counter()
            inference_fn()
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies.append(elapsed_ms)
            log.debug(f"Iter {i+1}: {elapsed_ms:.2f} ms")
        except Exception as e:
            error_msg = str(e)
            log.error(f"Benchmark iteration {i+1} failed: {e}")
            break

    mem_after = _measure_memory_mb()

    success = len(latencies) > 0

    if success:
        avg = statistics.mean(latencies)
        p50 = _compute_percentile(latencies, 50)
        p95 = _compute_percentile(latencies, 95)
        p99 = _compute_percentile(latencies, 99)
        throughput = 1000.0 / avg if avg > 0 else 0.0
        mem_delta = (mem_after - mem_before) if (mem_before and mem_after) else None

        log.info(f"Benchmark complete: avg={avg:.2f}ms, p95={p95:.2f}ms, throughput={throughput:.2f} req/s")

        return BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            runtime=runtime,
            device=device,
            provider=provider,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=len(latencies),
            latencies_ms=latencies,
            avg_latency_ms=avg,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            throughput_rps=throughput,
            memory_before_mb=mem_before,
            memory_after_mb=mem_after,
            memory_delta_mb=mem_delta,
            timestamp=datetime.now().isoformat(),
            success=True,
            notes=notes,
        )
    else:
        log.error(f"Benchmark failed: {error_msg}")
        return BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            runtime=runtime,
            device=device,
            provider=provider,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=0,
            latencies_ms=[],
            avg_latency_ms=0.0,
            min_latency_ms=0.0,
            max_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            throughput_rps=0.0,
            memory_before_mb=mem_before,
            memory_after_mb=mem_after,
            memory_delta_mb=None,
            timestamp=datetime.now().isoformat(),
            success=False,
            error_message=error_msg or "Unknown error",
            notes=notes,
        )


class BenchmarkSession:
    """Collects and compares benchmark results across devices."""

    def __init__(self):
        self._results: list[BenchmarkResult] = []

    def add(self, result: BenchmarkResult) -> None:
        self._results.append(result)

    def get_results(self) -> list[BenchmarkResult]:
        return self._results

    def compare(self) -> None:
        """Print a comparison table of all results."""
        if not self._results:
            print("No benchmark results to compare.")
            return

        header_div = "=" * 60
        sub_div = "-" * 55
        print(f"\n{header_div}")
        print("  SPECTRA PERFORMANCE LAB - COMPARISON")
        print(f"{header_div}")
        print(f"  {'Model':<25} {'Device':<8} {'Avg (ms)':<12} {'P95 (ms)':<12} {'Req/s':<10}")
        print(f"  {sub_div}")

        for r in self._results:
            if r.success:
                print(f"  {r.model_name:<25} {r.device:<8} "
                      f"{r.avg_latency_ms:<12.2f} {r.p95_latency_ms:<12.2f} {r.throughput_rps:<10.2f}")
            else:
                print(f"  {r.model_name:<25} {r.device:<8} {'NOT SUPPORTED':<35}")

        print(f"{header_div}\n")
