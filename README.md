# jax-cpu-gpu-tpu-final-project
# Churn Prediction: CPU vs GPU vs TPU/Multi-Node Scaling Benchmark

**Stanford University — Summer Session 2026**
**Course:** ME344 — Enterprise AI Systems Architecture & Infrastructure Scaling
**Student:** Michael Toms


---

## Overview

This repository contains the containerized training pipeline, orchestration manifests, and benchmarking material for the ME344 capstone project.

The project — "Scaling a Churn Prediction MLP Across CPU, GPU, and Multi-Node Infrastructure"— covers containerization, cluster orchestration, hardware-accelerated training, and performance benchmarking of a deep learning classifier across three distinct compute configurations.

The repository contains:

- a Dockerized MLP training pipeline for customer churn prediction;
- the dataset used throughout the experiments (`Churn_Dataset.csv`);
- Kubernetes/SLURM orchestration manifests specifying hardware constraints;
- benchmark results exported as JSON files for each hardware configuration;
- an aggregation notebook comparing CPU, GPU, and multi-node/TPU performance;
- the final presentation slides used for submission.

The main benchmarking component evaluates the **same MLP training workload** on three different hardware configurations — CPU, single GPU, and multi-node/TPU — under equivalent experimental conditions.

The objective is to provide a reproducible comparison of training performance while identifying the primary infrastructure bottleneck and the engineering mitigations that address it.

---

## Repository Structure

```
churn-mlp-scaling-benchmark/
│
├── docker/
│   └── Dockerfile
│
├── manifests/
│   ├── job_cpu.yaml            # or job_cpu.sbatch
│   ├── job_gpu.yaml            # or job_gpu.sbatch
│   └── job_multinode.yaml      # or job_tpu.sbatch
│
├── training/
│   ├── train.py                # single training script, hardware-agnostic
│   ├── model.py                # MLP definition
│   └── data_pipeline.py        # loading/preprocessing for Churn_Dataset.csv
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

The training pipeline is designed to run inside the Docker container built from `docker/Dockerfile`, so no manual dependency installation is required on any compute node.

Before running any benchmark:

1. Clone this repository.
2. Build the image: `docker build -t churn-mlp -f docker/Dockerfile .`
3. Confirm `dataset/Churn_Dataset.csv` is present (already included in the repo).

---

## Execution Workflow

Each hardware configuration runs the **same** `train.py` script, `model.py`, and `data_pipeline.py` — only the launch command and orchestration manifest change. This keeps the comparison fair: same code, same data, same preprocessing, same hyperparameters.

### Step 1 — CPU Benchmark

```
docker run churn-mlp python training/train.py --device cpu
```

Or via manifest: `manifests/job_cpu.yaml` (Kubernetes) / `job_cpu.sbatch` (SLURM).

Generates `cpu_metrics.json` inside `results/`.

### Step 2 — GPU Benchmark

```
docker run --gpus all churn-mlp python training/train.py --device gpu
```

Or via manifest: `manifests/job_gpu.yaml` / `job_gpu.sbatch`.

Generates `gpu_metrics.json` inside `results/`.

### Step 3 — Multi-Node / TPU Benchmark

Launched via `manifests/job_multinode.yaml` (GKE) or `job_tpu.sbatch` (HPCC), specifying the number of nodes/accelerators.

Generates `multinode_metrics.json` inside `results/`.

### Step 4 — Aggregate Results

Open `aggregation/Aggregation_Benchmarks.ipynb`. It:

- loads the three benchmark JSON files from `results/`;
- compares CPU, GPU, and multi-node/TPU performance;
- computes hardware speedups;
- produces the performance summary used in the README and slides.

---

## Benchmark Methodology

The benchmark measures steady-state training performance rather than one-time startup or compilation overhead.

- **[N] warm-up steps** performed before timing begins, to remove framework/JIT initialization overhead.
- **[N] measured training steps** collected per hardware configuration.
- Explicit synchronization before recording each timestamp (no measuring async dispatch as if it were compute time).
- Metrics captured per step: step time, throughput (samples/sec), hardware utilization %, peak memory usage.

Each configuration reports:

- Median step time
- Mean step time
- Minimum step time
- Standard deviation
- Peak memory / utilization

**Median is used as the primary comparison metric** (not mean), because shared cloud/cluster environments can produce occasional slow outlier steps from resource contention or scheduling — the median better reflects typical steady-state performance. Mean and standard deviation are still reported for a complete picture.

---

## Results

*(Fill in once your benchmark notebooks have run — this table drives your Slide 4.)*

| Hardware | Median Step Time | Throughput (samples/sec) | Utilization |
|---|---|---|---|
| CPU | [ ] ms | [ ] | — |
| GPU | [ ] ms | [ ] | [ ]% |
| Multi-Node/TPU | [ ] ms | [ ] | [ ]% |

Using the median as the reference metric, these measurements correspond to approximately:

- **[N]× speedup** from CPU → GPU
- **[N]× speedup** from CPU → Multi-Node/TPU
- **[N]× speedup** from GPU → Multi-Node/TPU

These results apply specifically to this MLP workload and dataset size and should not be interpreted as universal hardware performance ratios.

---

## Infrastructure Bottleneck Diagnosis

*(This becomes your README "bottleneck" section and Slide 5.)*

[State plainly, once you have data — e.g.: "At this dataset size, the model is too small to fully saturate the accelerators; step time is dominated by kernel-launch/dispatch overhead rather than raw compute, which is why speedup from GPU → multi-node is sub-linear."]

## Engineering Mitigations

- [e.g., increase batch size to better amortize per-step overhead]
- [e.g., mixed precision (bf16/fp16) to reduce memory pressure and increase throughput]
- [e.g., cache preprocessed data in memory instead of re-reading each epoch]
- [e.g., favor a single strong GPU over multi-node distribution for workloads this small]

---

## Engineering Takeaway

The purpose of this benchmark is not simply to determine which hardware is fastest in the abstract.

The results illustrate a broader systems engineering principle: **the optimal hardware and scaling strategy depend on the size and structure of the workload.** A small MLP on a modest tabular dataset does not automatically benefit from every additional device — coordination and dispatch overhead can offset the raw compute advantage of larger clusters unless the workload is large enough, or the implementation is structured carefully enough, to amortize that overhead.

Device availability is not the same as effective device utilization — performance should be evaluated in terms of how well the software maps the workload onto the hardware, not just which hardware is present.

---

## Reproducing This Benchmark

1. Clone this repository.
2. Build the Docker image (`docker/Dockerfile`).
3. Run each benchmark notebook/manifest for CPU, GPU, and multi-node/TPU (Steps 1–3 above).
4. Confirm all three JSON files exist in `results/`.
5. Run `aggregation/Aggregation_Benchmarks.ipynb` to produce the final comparison.

Because cloud hardware allocation, runtime configuration, and system load can vary between sessions, reproduced timing measurements may differ from the values reported here.
