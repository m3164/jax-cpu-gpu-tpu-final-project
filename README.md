# JAX Pairwise-Distance Benchmark: CPU vs GPU vs TPU

**Course:** Introduction to High Performance Computing
**Student:** Michael Toms

---

## Overview

This repository benchmarks a JAX-vectorized, JIT-compiled pairwise Euclidean distance kernel (the core of a K-Nearest-Neighbors churn classifier) across three hardware backends — CPU, GPU, and TPU — under equivalent experimental conditions.

The goal is to measure real execution-time deltas across hardware, quantify the actual speedup each accelerator provides for this workload, and identify what that speedup pattern implies about the kernel's bottleneck.

The repository contains:

- the K-Nearest-Neighbors churn-prediction pipeline (data loading, preprocessing, train/test split)
- the vectorized, `@jax.jit`-compiled pairwise-distance kernel
- the statistical benchmarking harness (warm-up + timed runs, full percentile spread)
- benchmark results for CPU, GPU, and TPU
- this README, aggregating and interpreting those results

---

## Repository Structure

```
jax-churn-knn-benchmark/
│
├── notebook/
│   └── Toms_Supervised_HPC_Assignment.ipynb
│
├── benchmarks/
│   ├── CPU_Benchmark.md
│   ├── GPU_Benchmark.md
│   └── TPU_Benchmark.md
│
├── dataset/
│   └── Churn_Dataset.csv
│
├── results/
│   ├── cpu_metrics.json
│   ├── gpu_metrics.json
│   └── tpu_metrics.json
│
├── README.md
└── requirements.txt
```

---

## Prerequisites

The benchmark notebook is designed to run in Google Colab, where CPU, GPU, and TPU runtimes can be selected independently (Runtime → Change runtime type).

1. Upload `Churn_Dataset.csv` to the Colab session.
2. Run the setup + preprocessing cells (Exercises 1–7).
3. Run the vectorized distance kernel and KNN prediction cells (Exercises 8–10).
4. Run the benchmark harness for each hardware target (Exercise 11 / `benchmark_backend()`).

---

## Execution Workflow

Colab exposes only one accelerator per session, so CPU, GPU, and TPU were benchmarked in three independent runs. Every run used:

- the same dataset, preprocessing pipeline, and train/test split
- the same JAX implementation of the pairwise-distance kernel
- the same benchmarking methodology (`benchmark_backend()`)

**Step 1 — CPU:** default Colab runtime, no switching needed → `cpu_metrics.json`
**Step 2 — GPU:** Runtime → GPU (T4) → `gpu_metrics.json`
**Step 3 — TPU:** Runtime → TPU → `tpu_metrics.json`

---

## Benchmark Methodology

- **10 warm-up iterations** per backend, to remove JIT/XLA compilation and initialization overhead before timing begins.
- **100 measured executions** per backend, timed individually with `time.perf_counter_ns()`.
- Explicit synchronization via `.block_until_ready()` before and after each timed call, so async dispatch is never counted as compute time.
- Each backend reports median, mean, min, P25, P75, and standard deviation.

**Median is used as the primary comparison metric.** Colab runs on shared cloud infrastructure, where occasional slow executions from resource contention or scheduling can pull the mean upward without reflecting typical performance. The median is far less sensitive to those outliers. Mean and standard deviation are still reported for a complete picture of the timing distribution.

---

## Results

| Hardware | Median | Mean | Min | P25 | P75 | Std Dev |
|---|---|---|---|---|---|---|
| CPU | 21.322 ms | 21.330 ms | 4.566 ms | 14.987 ms | 34.129 ms | 10.765 ms |
| GPU | 0.5001 ms | 0.5023 ms | 0.4765 ms | 0.4978 ms | 0.4998 ms | 0.0659 ms |
| TPU | 0.2111 ms | 0.2982 ms | 0.1438 ms | 0.2198 ms | 0.1976 ms | 0.0132 ms |

**Speedup, using the median as the reference metric:**

| Comparison | Speedup |
|---|---|
| CPU → GPU | ~42.6× |
| CPU → TPU | ~101.0× |
| GPU → TPU | ~2.4× |

---

## Infrastructure Bottleneck Diagnosis

The dominant speedup in this benchmark comes from CPU → GPU (~42.6×), not from GPU → TPU (~2.4×). This pattern is consistent with a workload that is fundamentally matrix-multiplication-bound: the pairwise-distance kernel reduces to `A² - 2AB + B²`, a dense matmul-heavy operation that both GPUs and TPUs are purpose-built to accelerate. The much smaller marginal gain from GPU → TPU suggests that, at this dataset's size, the kernel already captures most of what parallel hardware can offer — TPU's systolic-array design pulls further ahead, but there isn't a second CPU-to-accelerator-scale gap left to close.

A second, independently notable finding: CPU execution time varies far more than GPU or TPU (std dev 10.765 ms vs. 0.0659 ms and 0.0132 ms respectively; P25–P75 spread of roughly 19 ms on CPU vs. under 0.03 ms on the accelerators). This is consistent with CPU execution being more exposed to scheduling and resource contention on shared Colab infrastructure — a concrete, data-backed justification for using median over mean as this project's primary metric.

## Engineering Mitigations / Recommendations

- For workloads dominated by dense matrix operations like this one, prioritize GPU/TPU allocation over CPU — the CPU → GPU jump captures the overwhelming majority of the available speedup.
- Given the small marginal GPU → TPU gain here, GPU is likely the better cost/performance choice unless the workload scales up significantly (larger batch sizes, larger feature dimensionality) to where TPU's architecture has more headroom to exploit.
- Because CPU timing is noisy, any CPU-based benchmark comparison should always report median (or a trimmed statistic) rather than mean, and should run enough iterations to get a stable percentile spread.

---

## Reproducing This Benchmark

1. Open the notebook in Colab.
2. Upload `Churn_Dataset.csv`.
3. Run Exercises 1–10 to prepare data and compile the kernel.
4. Run `benchmark_backend()` for `backend="cpu"`.
5. Switch Runtime → GPU, re-run setup cells, run `benchmark_backend()` for `backend="gpu"`.
6. Switch Runtime → TPU, re-run setup cells, run `benchmark_backend()` for `backend="tpu"`.
7. Save each result to its respective JSON file in `results/`.

Because Colab hardware allocation and system load vary between sessions, reproduced timings may differ slightly from the values reported here.
