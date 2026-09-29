# SPECTRA Performance Lab & Benchmarking Methodology

## 1. Zero Fabrication Policy

In strict accordance with the competition objectives:
* **All benchmarks are executed live on physical hardware.**
* If an execution provider (such as QNN or CUDA) is unavailable, the device is marked **`[UNAVAILABLE]`** and excluded from synthetic calculations.
* Benchmarks employ high-precision monotonic timing (`time.perf_counter()`).

---

## 2. Profiling Pipeline

```text
Warmup Iterations (JIT / Memory Allocation)
                    |
                    v
Inference Loop (10-50 iterations)
                    |
                    v
Monotonic Clock Timing: Delta = (t_end - t_start) * 1000.0 ms
                    |
                    v
Statistical Computation:
- Mean Latency
- Min / Max Bounds
- P50 (Median)
- P95 (95th Percentile)
- P99 (Worst Case)
- Throughput (1000.0 / Mean Latency)
- Working Set Delta (RSS memory before vs after)
```

---

## 3. Real Benchmark Results (Development Host Machine)

```text
============================================================
  SPECTRA PERFORMANCE LAB - COMPARISON
============================================================
  Model                     Device   Avg (ms)     P95 (ms)     Req/s     
  -------------------------------------------------------
  MobileNetV2 (ONNX)        CPU      6.95         9.38         143.88    
============================================================
```

* **Warmup**: 3 iterations  
* **Benchmark iterations**: 15  
* **Min Latency**: 5.07 ms  
* **Max Latency**: 9.82 ms  
* **Memory Delta**: +10.7 MB  
