"""
Analyze how often neighboring genes intrude into the 6kb windows
used by the AgroNT seq2expression benchmark.

For each gene in the training/test CSVs, we:
1. Look up its genomic coordinates from the GFF3
2. Compute the 6kb window (5kb upstream + 1kb downstream of gene start)
3. Check if any other genes overlap that window
4. Report how many bp of the window are "contaminated" by other genes
"""

import gzip
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

RAW_DATA_DIR = "/nfs/hpc/share/evo2_shared/Seq2ExpBenchmarking/data/raw_bp"
SEQ2EXP_DIR = "/nfs/hpc/share/evo2_shared/Seq2ExpBenchmarking/data/seq2exp"
OUTPUT_DIR = "/nfs/hpc/share/minchle/projects/CS46X_Project/analysis/outputs"

SPECIES = {
    "arabidopsis_thaliana": "A. thaliana",
    "glycine_max": "G. max",
    "oryza_sativa": "O. sativa",
    "solanum_lycopersicum": "S. lycopersicum",
    "zea_mays": "Z. mays",
}


def parse_genes_from_gff3(gff3_path):
    """
    Parse all gene features from GFF3.
    Returns dict: gene_id -> {chr, start, end, strand}
    Also returns a list of (chr, start, end, strand, gene_id) for interval queries.
    """
    genes = {}
    gene_list = []
    with gzip.open(gff3_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue

            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]

            attrs = fields[8]
            gene_id = None
            for attr in attrs.split(";"):
                if attr.startswith("ID=gene:"):
                    gene_id = attr.split("ID=gene:")[1]
                    break
            if gene_id:
                genes[gene_id] = {
                    "chr": chrom, "start": start, "end": end, "strand": strand
                }
                gene_list.append((chrom, start, end, strand, gene_id))

    return genes, gene_list


def get_gene_ids_from_csv(species_dir):
    """Get all gene IDs from the train/test/validation CSVs."""
    gene_ids = set()
    for split in ["train", "test", "validation"]:
        csv_path = os.path.join(SEQ2EXP_DIR, species_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gene_ids.add(row["name"])
    return gene_ids


def build_chr_gene_index(gene_list):
    """Build a per-chromosome sorted list of genes for fast overlap queries."""
    chr_genes = defaultdict(list)
    for chrom, start, end, strand, gene_id in gene_list:
        chr_genes[chrom].append((start, end, gene_id))
    # Sort by start position
    for chrom in chr_genes:
        chr_genes[chrom].sort()
    return chr_genes


def find_overlapping_genes(chr_genes, chrom, win_start, win_end, target_gene_id):
    """Find all genes on `chrom` that overlap [win_start, win_end], excluding target."""
    overlaps = []
    for gstart, gend, gid in chr_genes.get(chrom, []):
        if gstart > win_end:
            break
        if gend < win_start:
            continue
        if gid == target_gene_id:
            continue
        # Overlap region within the window
        ovl_start = max(gstart, win_start)
        ovl_end = min(gend, win_end)
        overlaps.append((ovl_start, ovl_end, gid))
    return overlaps


def compute_intrusion_bp(overlaps, win_start, win_end):
    """Compute total bp of intrusion (merging overlapping regions)."""
    if not overlaps:
        return 0
    # Merge overlapping intervals
    intervals = sorted([(s, e) for s, e, _ in overlaps])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return sum(e - s + 1 for s, e in merged)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for species_dir, display_name in SPECIES.items():
        print(f"\n{'='*60}")
        print(f"Processing {display_name}...")
        print(f"{'='*60}")

        # Parse GFF3
        gff3_files = [f for f in os.listdir(os.path.join(RAW_DATA_DIR, species_dir))
                       if f.endswith(".gff3.gz")]
        gff3_path = os.path.join(RAW_DATA_DIR, species_dir, gff3_files[0])
        genes, gene_list = parse_genes_from_gff3(gff3_path)
        chr_genes = build_chr_gene_index(gene_list)

        # Get benchmark gene IDs
        benchmark_ids = get_gene_ids_from_csv(species_dir)
        matched = benchmark_ids & set(genes.keys())
        print(f"  Benchmark genes: {len(benchmark_ids)}, matched to GFF3: {len(matched)}")

        # Analyze each benchmark gene
        intrusion_bps = []
        num_neighbors = []
        intrusion_fractions = []
        has_intrusion = 0
        upstream_intrusions = 0
        downstream_intrusions = 0

        for gene_id in sorted(matched):
            g = genes[gene_id]
            # Compute 6kb window based on strand
            if g["strand"] == "+":
                tss = g["start"]
                win_start = tss - 5000
                win_end = tss + 999
            else:
                tss = g["end"]
                win_start = tss - 999
                win_end = tss + 5000

            overlaps = find_overlapping_genes(chr_genes, g["chr"], win_start, win_end, gene_id)
            bp = compute_intrusion_bp(overlaps, win_start, win_end)

            intrusion_bps.append(bp)
            num_neighbors.append(len(overlaps))
            intrusion_fractions.append(bp / 6000)

            if bp > 0:
                has_intrusion += 1
                for ovl_start, ovl_end, _ in overlaps:
                    if g["strand"] == "+":
                        if ovl_start < tss:
                            upstream_intrusions += 1
                        if ovl_end >= tss:
                            downstream_intrusions += 1
                    else:
                        if ovl_end > tss:
                            upstream_intrusions += 1
                        if ovl_start <= tss:
                            downstream_intrusions += 1

        intrusion_bps = np.array(intrusion_bps)
        num_neighbors = np.array(num_neighbors)
        intrusion_fractions = np.array(intrusion_fractions)

        pct_with = 100 * has_intrusion / len(matched)
        print(f"  Genes with neighbor intrusion: {has_intrusion}/{len(matched)} ({pct_with:.1f}%)")
        print(f"  Among those with intrusion:")
        if has_intrusion > 0:
            intruded = intrusion_bps[intrusion_bps > 0]
            print(f"    Mean intrusion: {np.mean(intruded):.0f} bp")
            print(f"    Median intrusion: {np.median(intruded):.0f} bp")
            print(f"    Max intrusion: {np.max(intruded):.0f} bp")
            print(f"    Mean % of window: {100*np.mean(intruded/6000):.1f}%")
        print(f"  Upstream neighbor intrusions: {upstream_intrusions}")
        print(f"  Downstream neighbor intrusions: {downstream_intrusions}")
        print(f"  Mean neighbors per window: {np.mean(num_neighbors):.2f}")

        all_results[display_name] = {
            "intrusion_bps": intrusion_bps,
            "num_neighbors": num_neighbors,
            "intrusion_fractions": intrusion_fractions,
            "pct_with": pct_with,
            "n_genes": len(matched),
        }

    # --- Plot 1: % of genes with intrusion per species ---
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(all_results.keys())
    pcts = [all_results[n]["pct_with"] for n in names]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(names, pcts, color=colors, alpha=0.7)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("% of benchmark genes")
    ax.set_title("Genes with Neighboring Gene Intrusion in 6kb Window")
    ax.set_ylim(0, max(pcts) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "neighbor_intrusion_pct.png"), dpi=150)
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'neighbor_intrusion_pct.png')}")

    # --- Plot 2: Distribution of intrusion bp (only genes with intrusion) ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    for i, (name, res) in enumerate(all_results.items()):
        ax = axes[i]
        intruded = res["intrusion_bps"][res["intrusion_bps"] > 0]
        if len(intruded) > 0:
            ax.hist(intruded, bins=50, range=(0, 6000), color=colors[i], alpha=0.7)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Intrusion (bp)")
        if i == 0:
            ax.set_ylabel("Count")
    fig.suptitle("Distribution of Neighbor Gene Intrusion (bp) — genes with intrusion only", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "neighbor_intrusion_dist.png"), dpi=150)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'neighbor_intrusion_dist.png')}")

    # --- Plot 3: Number of neighbor genes per window ---
    fig, ax = plt.subplots(figsize=(10, 5))
    n_bins = 11  # 0 through 10+
    x = np.arange(n_bins)
    width = 0.15
    for i, (name, res) in enumerate(all_results.items()):
        clamped = np.minimum(res["num_neighbors"], 10)
        counts = np.bincount(clamped, minlength=n_bins)[:n_bins]
        pcts = 100 * counts / res["n_genes"]
        ax.bar(x + i * width, pcts, width, label=name, color=colors[i], alpha=0.7)
    ax.set_xlabel("Number of neighboring genes in window")
    ax.set_ylabel("% of benchmark genes")
    ax.set_title("Number of Neighboring Genes Overlapping 6kb Window")
    ax.set_xticks(x + 2 * width)
    xlabels = [str(i) for i in range(10)] + ["10+"]
    ax.set_xticklabels(xlabels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "neighbor_count_dist.png"), dpi=150)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'neighbor_count_dist.png')}")


if __name__ == "__main__":
    main()
