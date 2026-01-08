# Helixer Integration Documentation

This document explains the setup, usage, and capabilities of the Helixer gene prediction integration.

## 1. Environment Setup

The Helixer integration requires a specific environment to handle dependencies like `tensorflow-addons`. We have created a dedicated Conda environment for this purpose.

### Using the Existing Environment
The environment is already set up and ready to use.

**Environment Name**: `helixer_env`
**Python Path**: `/nfs/stak/users/minchle/miniconda3/envs/helixer_env/bin/python`

**To Activate and Run:**
```bash
# Option 1: Activate globally
conda activate helixer_env
python helixer_runner.py ...

# Option 2: Use direct path (Recommended for scripts/SLURM)
/nfs/stak/users/minchle/miniconda3/envs/helixer_env/bin/python helixer_runner.py ...
```

### Recreating the Environment (Reference)
If you need to recreate this environment on another machine, follow these steps:
1.  **Create Conda Env (Python 3.10 is required)**:
    ```bash
    conda create -n helixer_env python=3.10
    conda activate helixer_env
    ```
2.  **Install Dependencies**:
    ```bash
    pip install helixerlite biopython
    ```

---

## 2. Capabilities

The integration is driven by the `helixer_runner.py` wrapper script, which enhances the raw `helixerlite` tool with several features robust enough for production pipelines.

### A. Core Gene Prediction
You can run gene prediction on any FASTA file. The script uses the `helixerlite` inference engine (default model: `land_plant`).

**Command:**
```bash
python helixer_runner.py --fasta <genome.fasta> --output <results.gff>
```

### B. Targeted Region Analysis
If you only want to analyze specific regions (e.g., candidate loci identified by another tool), you can pass a GFF file defining those regions. The script will extract only those sequences and predict coverage for them, saving computational resources.

**Command:**
```bash
python helixer_runner.py --gff <regions_of_interest.gff> --fasta <genome.fasta> --output <results.gff>
```

### C. Robust Input Sanitization (Key Feature)
Standard Helixer crashes on "dirty" data. Our wrapper automatically handles:
*   **RNA Input**: Automatically converts Uracil (`U`) to Thymine (`T`).
*   **Invalid Characters**: Strips or masks non-standard nucleotides (e.g., `P`, spaces, tabs) that commonly appear in raw datasets.
*   **Garbage Metadata**: Filters out hidden system files (like macOS `._` resource forks).

### D. Stability & Resource Management
*   **Thread Safety**: Automatically configures TensorFlow/OMP threading environment variables to prevent segmentation faults (`SIGABRT`) on shared HPC nodes.

---

## 3. Associated Files
*   **`helixer_runner.py`**: The main execution script described above.
*   **`benchmark_helixer.py`**: A utility to calculate sensitivity/overlap between Helixer predictions and a ground truth GFF (validated on TAIR10 Mitochondria).
*   **`future_implementations.md`**: Roadmap for integrating with **Evo2** and training specialized grain models.
