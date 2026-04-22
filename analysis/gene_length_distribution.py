"""
Gene length distribution analysis across all 5 species in the PGB benchmark.
Extracts gene features from GFF3 files and plots length distributions.
"""

import gzip
import os
import numpy as np
import matplotlib.pyplot as plt

RAW_DATA_DIR = "/nfs/hpc/share/evo2_shared/Seq2ExpBenchmarking/data/raw_bp"
OUTPUT_DIR = "/nfs/hpc/share/minchle/projects/CS46X_Project/analysis/outputs"

SPECIES = {
    "arabidopsis_thaliana": "A. thaliana",
    "glycine_max": "G. max",
    "oryza_sativa": "O. sativa",
    "solanum_lycopersicum": "S. lycopersicum",
    "zea_mays": "Z. mays",
}


def parse_gene_lengths(gff3_path):
    """Extract gene lengths from a GFF3 file. Returns dict of gene_id -> length."""
    genes = {}
    with gzip.open(gff3_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != "gene":
                continue

            start = int(fields[3])
            end = int(fields[4])
            length = end - start + 1

            # Parse gene ID from attributes
            attrs = fields[8]
            gene_id = None
            for attr in attrs.split(";"):
                if attr.startswith("ID=gene:"):
                    gene_id = attr.split("ID=gene:")[1]
                    break
            if gene_id:
                genes[gene_id] = length

    return genes


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_species_data = {}

    for species_dir, display_name in SPECIES.items():
        species_path = os.path.join(RAW_DATA_DIR, species_dir)
        gff3_files = [f for f in os.listdir(species_path) if f.endswith(".gff3.gz")]
        if not gff3_files:
            print(f"No GFF3 found for {species_dir}, skipping.")
            continue

        gff3_path = os.path.join(species_path, gff3_files[0])
        print(f"Parsing {display_name} from {gff3_files[0]}...")
        genes = parse_gene_lengths(gff3_path)
        lengths = np.array(list(genes.values()))
        all_species_data[display_name] = lengths

        print(f"  {len(lengths):,} genes")
        print(f"  Mean:   {np.mean(lengths):,.0f} bp")
        print(f"  Median: {np.median(lengths):,.0f} bp")
        print(f"  Min:    {np.min(lengths):,} bp")
        print(f"  Max:    {np.max(lengths):,} bp")
        print(f"  Std:    {np.std(lengths):,.0f} bp")
        print(f"  % > 6kb: {100 * np.mean(lengths > 6000):.1f}%")
        print()

    # --- Plot 1: Overlaid histograms ---
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, lengths) in enumerate(all_species_data.items()):
        ax.hist(lengths, bins=100, range=(0, 20000), alpha=0.5,
                label=f"{name} (n={len(lengths):,})", color=colors[i])

    ax.axvline(x=6000, color="black", linestyle="--", linewidth=1.5, label="6kb window")
    ax.set_xlabel("Gene Length (bp)")
    ax.set_ylabel("Count")
    ax.set_title("Gene Length Distribution by Species")
    ax.legend()
    ax.set_xlim(0, 20000)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gene_length_hist_overlay.png"), dpi=150)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'gene_length_hist_overlay.png')}")

    # --- Plot 2: Box plots side by side ---
    fig, ax = plt.subplots(figsize=(10, 6))
    data_for_box = [lengths for lengths in all_species_data.values()]
    labels = list(all_species_data.keys())
    bp = ax.boxplot(data_for_box, labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=6000, color="black", linestyle="--", linewidth=1.5, label="6kb window")
    ax.set_ylabel("Gene Length (bp)")
    ax.set_title("Gene Length Distribution by Species (outliers hidden)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gene_length_boxplot.png"), dpi=150)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'gene_length_boxplot.png')}")

    # --- Plot 3: Per-species subplots with log scale ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for i, (name, lengths) in enumerate(all_species_data.items()):
        ax = axes[i]
        ax.hist(lengths, bins=80, range=(0, 30000), color=colors[i], alpha=0.7)
        ax.axvline(x=6000, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("bp")
        if i == 0:
            ax.set_ylabel("Count")
        ax.set_yscale("log")

    fig.suptitle("Gene Length Distributions (log scale)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gene_length_per_species.png"), dpi=150)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'gene_length_per_species.png')}")


if __name__ == "__main__":
    main()
