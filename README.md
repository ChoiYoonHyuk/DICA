# DICA

Research code for **Exact Sampling from KV Summaries: Access Limits and Residual Criteria**.

## Overview

We study when key-value (KV) summaries support exact next-token sampling before all attention values are retrieved.

- **Access limits:** sharp read–error trade-offs for changing top-k support and probability-weighted costs for isolated cutoff pairs.
- **Residual sampling:** exact two-group full-softmax sampling using attention directions left unexplained by a summary.
- **Certification and cost:** analysis of probability certification, vocabulary normalization, value reads, memory traffic, and runtime.

The paper includes synthetic studies and experiments with Qwen3-4B and Gemma4-E4B on FineWeb and LongBench.

## Quick start

```bash
git clone https://github.com/ChoiYoonHyuk/DICA.git
cd DICA
python -m pip install numpy torch
python corrected_runner.py --self-test
```

The self-test runs CPU numerical checks without downloading model weights or datasets.

## Model experiments

Use a CUDA-enabled PyTorch environment with BF16 support. Install the additional dependencies below, choosing a Transformers release that supports your checkpoint:

```bash
python -m pip install "transformers>=4.57.1,<6" datasets \
  huggingface_hub "pyarrow>=25.0.0" triton

python corrected_runner.py --models qwen --suite smoke --datasets fineweb \
  --out results/smoke --no-task-generation
```

Use `--models gemma` for Gemma4, or pass a checkpoint path. Model checkpoints and datasets are downloaded separately and may require Hugging Face access.

The complete-decoder benchmark requires a prepared evaluation plan and its referenced token files, which are not bundled in this repository:

```bash
python main.py --family qwen3 --plan /path/to/plan.json \
  --output results/generation
```

Plan entries specify the model checkpoint. Use `--family gemma4` for Gemma4 and `--help` to inspect each runner's options.

## Main files

| File | Purpose |
| --- | --- |
| [main.py](main.py), [host_generation_benchmark.py](host_generation_benchmark.py) | Complete-decoder benchmark with shared CPU embedding and output weights. |
| [corrected_runner.py](corrected_runner.py) | KV-access experiments, certification, and numerical checks. |
| [host_projection.py](host_projection.py), [host_projection_benchmark.py](host_projection_benchmark.py) | Host-source output-projection implementation and benchmark. |
| [exact_grid_sampler.py](exact_grid_sampler.py), [suffix_refined.py](suffix_refined.py) | Finite-grid sampling and finite-image suffix certificates. |
| `policy_projection*.py`, [packed_topk.py](packed_topk.py) | Projection bounds, Triton kernels, and deterministic top-k selection. |

## Scope

Numerical exactness is defined relative to the specified reference computation. Reduced reads or memory transfers alone do not guarantee faster end-to-end decoding; the experiments report these costs separately.

