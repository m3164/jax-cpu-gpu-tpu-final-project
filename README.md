# Churn Prediction: CPU vs GPU vs TPU/Multi-Node Scaling Benchmark

**Stanford University — Summer Session 2026**
**Course:** ME344 — Enterprise AI Systems Architecture & Infrastructure Scaling
**Student:** Michael Toms


---

## Overview

This repository contains the containerized training pipeline, orchestration manifests, and benchmarking material for the ME344 capstone project.

The project evaluates a Multi-Layer Perceptron (MLP) churn classifier's training performance across three compute configurations — CPU, single GPU, and multi-node/TPU — under equivalent experimental conditions.

The repository contains:

- a Dockerized MLP training pipeline for customer churn prediction;
- the dataset used throughout the experiments (`Churn_Dataset.csv`);
- Kubernetes/SLURM orchestration manifests specifying hardware constraints;
- benchmark results exported as JSON files for each hardware configuration;
- an aggregation notebook comparing CPU, GPU, and multi-node/TPU performance;
- the final presentation slides used for submission.

---

## Repository Structure

```
churn-mlp-scaling-benchmark/
│
├── docker/
│   └── Dockerfile
│
├── manifests/
│   ├── job_cpu.yaml
│   ├── job_gpu.yaml
│   └── job_multinode.yaml
│
├── training/
│   ├── train.py
│   ├── model.py
│   └── data_pipeline.py
│
├── benchmarks/
│   ├── CPU_Benchmark.ipynb
│   ├── GPU_Benchmark.ipynb
│   └── MultiNode_TPU_Benchmark.ipynb
│
├── aggregation/
│   └── Aggregation_Benchmarks.ipynb
│
├── dataset/
│   └── Churn_Dataset.csv
│
├── results/
│   ├── cpu_metrics.json
│   ├── gpu_metrics.json
│   └── multinode_metrics.json
│
├── presentation/
│   └── ME344_Capstone_[YourName].pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Prerequisites

The training pipeline runs inside the Docker container built from `docker/Dockerfile`, so no manual dependency installation is required on any compute node.

1. Clone this repository.
2. Build the image: `docker build -t churn-mlp -f docker/Dockerfile .`
3. Confirm `dataset/Churn_Dataset.csv` is present.

---

## Execution Workflow

Each hardware configuration runs the same `train.py`, `model.py`, and `data_pipeline.py` — only the launch command and orchestration manifest change, keeping the comparison fair.

**Step 1 — CPU:** `docker run churn-mlp python training/train.py --device cpu` → generates `cpu_metrics.json`
**Step 2 — GPU:** `docker run --gpus all churn-mlp python training/train.py --device gpu` → generates `gpu_metrics.json`
**Step 3 — Multi-Node/TPU:** launched via `manifests/job_multinode.yaml` → generates `multinode_metrics.json`
**Step 4 — Aggregate:** `aggregation/Aggregation_Benchmarks.ipynb` loads all three JSON files and computes speedups.

---

## Benchmark Methodology

*(EXAMPLE VALUES — replace with your actual run configuration)*

- **10 warm-up steps** performed before timing begins, to remove framework/JIT initialization overhead.
- **100 measured training steps** collected per hardware configuration.
- Explicit synchronization before recording each timestamp.
- Metrics captured per step: step time, throughput (samples/sec), hardware utilization %, peak memory usage.

**Median is used as the primary comparison metric** rather than mean, since shared cloud/cluster environments occasionally produce slow outlier steps from resource contention or scheduling — median better reflects typical steady-state performance. Mean and standard deviation are still reported for a complete picture.

---

## Results -

| Hardware | Median Step Time | Throughput (samples/sec) | Utilization |
|---|---|---|---|
| CPU | 41.6 ms | ~1,540 | — |
| GPU | 3.2 ms | ~20,000 | ~62% |
| Multi-Node/TPU (4×) | 1.1 ms | ~58,000 | ~41% per device |

Using the median as the reference metric, these placeholder measurements correspond to approximately:

- **~13× speedup** from CPU → GPU
- **~38× speedup** from CPU → Multi-Node/TPU
- **~2.9× speedup** from GPU → Multi-Node/TPU (well below the 4× a 4-device pool would give under perfect linear scaling)

---

## Infrastructure Bottleneck Diagnosis — 

"The GPU→multi-node speedup (~2.9×) falls well short of the 4× a 4-device pool would give under perfect linear scaling. With a model and batch this small, per-step compute finishes in a couple of milliseconds — too little useful work to fully amortize the dispatch and synchronization overhead of coordinating multiple devices, so a meaningful share of each multi-node step is coordination, not computation."

## Engineering Mitigations — 

- Increase batch size to give each device more work per synchronization point.
- Use mixed precision (bf16/fp16) to reduce memory pressure and increase throughput.
- Cache preprocessed data in memory instead of re-reading/re-transforming each epoch.
- For a workload this small, a single strong GPU may be the more cost-effective choice over multi-node distribution.

---

## Engineering Takeaway

The purpose of this benchmark is not simply to determine which hardware is fastest in the abstract. The results should illustrate that the optimal hardware and scaling strategy depend on the size and structure of the workload — a small MLP on a modest tabular dataset does not automatically benefit from every additional device, since coordination overhead can offset the raw compute advantage of a larger cluster unless the workload is large enough to amortize it.

---

## Reproducing This Benchmark

1. Clone this repository.
2. Build the Docker image (`docker/Dockerfile`).
3. Run each benchmark notebook/manifest for CPU, GPU, and multi-node/TPU.
4. Confirm all three JSON files exist in `results/`.
5. Run `aggregation/Aggregation_Benchmarks.ipynb` to produce the final comparison.

Because cloud hardware allocation, runtime configuration, and system load can vary between sessions, your measured timings will differ from the illustrative values above — that's expected and fine, since those values aren't real to begin with.
