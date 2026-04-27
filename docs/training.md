---
layout: default
title: Training Guide
---

# Training Guide

[← Back to Home](index.md)

---

## HPC Setup

Training runs on the Oregon State University HPC cluster using SLURM.

### Environment

```bash
module load cuda/12.8
module load cudnn/8.9_cuda12
conda activate evo2
```

### Configuration

Training configurations are stored in `configs/`. Each config specifies:

- Model checkpoint path
- Dataset paths (`training_data/`)
- Batch size, learning rate, and scheduler
- Output and checkpoint directories (`checkpoints/`)

---

## Running Training

*(Details to be added as training scripts are finalized.)*

---

## Checkpoints

Model checkpoints are saved to `checkpoints/` and should **not** be committed to git (see `.gitignore`). Reference checkpoint paths in experiment logs with dates and run IDs for reproducibility.

---

## Logging & Reproducibility

- All experiment results, configurations, and outputs are logged with references and dates per [CONTRIBUTING.md](contributing.md)
- Include lockfiles and metrics when tagging releases
