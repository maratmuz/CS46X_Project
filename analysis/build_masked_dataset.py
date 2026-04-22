"""
Build a masked version of the seq2exp dataset.

For each gene's 6kb sequence, identify neighboring genes that intrude
into the window and replace their bases with AgroNT's ``<mask>`` special
token on 6bp-aligned boundaries. This preserves the target gene's
promoter region and gene body while removing information from unrelated
neighboring genes using the model's own masking semantics.

Why ``<mask>`` and not 'N'? AgroNT's 6-mer tokenizer has no N-containing
6-mer tokens, so any N forces fallback to one-token-per-character.
Masking ~3kb of a 6kb window with N produces up to ~5000 tokens, far
exceeding AgroNT's 1024-token position-embedding limit and causing
out-of-bounds CUDA asserts. The literal string ``<mask>`` is exactly
6 characters and tokenizes to a single special mask token, so when we
substitute it for a 6bp block the 6bp-per-token ratio is preserved,
positional alignment with flanking real DNA is maintained, and the model
sees the same mask token it was pretrained on.

Neighbor-gene spans are snapped outward to the nearest 6bp boundaries so
every masked block starts on a tokenization boundary. This can slightly
over-mask (up to 5bp per edge) but keeps tokens aligned.

Output format matches the original seq2exp CSVs:
    name, sequence, labels, split

The ``sequence`` column is still 6000 characters but mixes DNA with
``<mask>`` literals (each 6 chars wide).
"""

MASK_TOKEN = "<mask>"  # AgroNT's special mask token; 6 chars, tokenizes as 1 token
BLOCK = 6              # AgroNT's k-mer size; all masks snap to this boundary
assert len(MASK_TOKEN) == BLOCK

import gzip
import csv
import os
import json
import subprocess
import numpy as np
from collections import defaultdict


def robust_listdir(path):
    """Workaround for flaky NFS permissions on this cluster where
    os.listdir/glob/Path.iterdir sporadically raise EACCES even though
    the user is in the owning group. subprocess ls is consistently
    reliable on the same paths."""
    return subprocess.check_output(["ls", path], text=True).split()

RAW_DATA_DIR = "/nfs/hpc/share/evo2_shared/Seq2ExpBenchmarking/data/raw_bp"
SEQ2EXP_DIR = "/nfs/hpc/share/evo2_shared/Seq2ExpBenchmarking/data/seq2exp"
OUTPUT_DIR = "/nfs/hpc/share/evo2_shared/Seq2ExpBenchmarking/data/seq2expMasked"

SPECIES = [
    "arabidopsis_thaliana",
    "glycine_max",
    "oryza_sativa",
    "solanum_lycopersicum",
    "zea_mays",
]


def parse_genes_from_gff3(gff3_path):
    """Parse all gene features from GFF3.
    Returns:
        genes: dict of gene_id -> {chr, start, end, strand}
        gene_list: list of (chr, start, end, strand, gene_id)
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


def build_chr_gene_index(gene_list):
    """Build a per-chromosome sorted list of genes for overlap queries."""
    chr_genes = defaultdict(list)
    for chrom, start, end, strand, gene_id in gene_list:
        chr_genes[chrom].append((start, end, gene_id))
    for chrom in chr_genes:
        chr_genes[chrom].sort()
    return chr_genes


def find_overlapping_genes(chr_genes, chrom, win_start, win_end, target_gene_id):
    """Find all genes on chrom that overlap [win_start, win_end], excluding target."""
    overlaps = []
    for gstart, gend, gid in chr_genes.get(chrom, []):
        if gstart > win_end:
            break
        if gend < win_start:
            continue
        if gid == target_gene_id:
            continue
        overlaps.append((gstart, gend, gid))
    return overlaps


def genomic_to_seq_position(genomic_pos, win_start, win_end, strand):
    """Convert a genomic position to a sequence position (0-indexed).

    For + strand: sequence reads left-to-right in genomic coords.
        seq_pos = genomic_pos - win_start

    For - strand: sequence is reverse-complemented, so position 0 in the
        sequence corresponds to the rightmost genomic position.
        seq_pos = win_end - genomic_pos
    """
    if strand == "+":
        return genomic_pos - win_start
    else:
        return win_end - genomic_pos


def sanitize_ns(sequence):
    """Replace any 6bp-aligned block containing at least one N with ``<mask>``.

    Plant-genomic-benchmark source sequences contain N's from assembly gaps,
    which — like our own masking — would otherwise fall back to per-character
    tokenization and blow past AgroNT's 1024-token limit. This pass guarantees
    every 6bp block is either pure ACGT (1 DNA 6-mer token) or exactly
    ``<mask>`` (1 mask token).
    """
    seq_list = list(sequence)
    for b in range(0, len(sequence) - BLOCK + 1, BLOCK):
        if "N" in sequence[b:b + BLOCK]:
            for k in range(BLOCK):
                seq_list[b + k] = MASK_TOKEN[k]
    return "".join(seq_list)


def mask_sequence(sequence, gene_info, chr_genes):
    """Mask neighboring gene regions in a 6kb sequence with ``<mask>`` tokens.

    Args:
        sequence: the 6kb DNA string
        gene_info: dict with chr, start, end, strand for the target gene
        chr_genes: per-chromosome sorted gene index

    Returns:
        masked_sequence: string with neighbor 6bp-aligned blocks replaced
            by the literal 6-char string ``<mask>``
        n_masked_bp: number of original bases overwritten
    """
    strand = gene_info["strand"]

    # Compute the 6kb window in genomic coordinates
    if strand == "+":
        tss = gene_info["start"]
        win_start = tss - 5000
        win_end = tss + 999
    else:
        tss = gene_info["end"]
        win_start = tss - 999
        win_end = tss + 5000

    # Find neighboring genes that overlap the window
    overlaps = find_overlapping_genes(
        chr_genes, gene_info["chr"], win_start, win_end, None
    )

    # Remove the target gene itself from overlaps
    target_id_start = gene_info["start"]
    target_id_end = gene_info["end"]
    overlaps = [
        (gs, ge, gid) for gs, ge, gid in overlaps
        if not (gs == target_id_start and ge == target_id_end)
    ]

    if not overlaps:
        return sequence, 0

    # Convert sequence to a mutable list
    seq_list = list(sequence)
    n_masked = 0

    for gs, ge, gid in overlaps:
        # Clamp neighbor gene to the window boundaries
        ovl_start = max(gs, win_start)
        ovl_end = min(ge, win_end)

        # Convert to sequence positions
        if strand == "+":
            seq_start = ovl_start - win_start
            seq_end = ovl_end - win_start
        else:
            # Reverse complement flips the coordinate order
            seq_start = win_end - ovl_end
            seq_end = win_end - ovl_start

        # Clamp to valid sequence range
        seq_start = max(0, seq_start)
        seq_end = min(len(sequence) - 1, seq_end)

        # Snap outward to 6bp boundaries so every masked block aligns to
        # AgroNT's 6-mer tokenization grid. Within each block, write the
        # six characters of "<mask>" so the tokenizer emits one <mask> token
        # per 6bp — matching the 6bp/token rate of the surrounding real DNA.
        block_start = (seq_start // BLOCK) * BLOCK
        block_end_excl = ((seq_end + BLOCK) // BLOCK) * BLOCK
        for b in range(block_start, min(block_end_excl, len(sequence) - BLOCK + 1), BLOCK):
            for k in range(BLOCK):
                if seq_list[b + k] != MASK_TOKEN[k]:
                    seq_list[b + k] = MASK_TOKEN[k]
                    n_masked += 1

    return "".join(seq_list), n_masked


def process_species(species_dir):
    """Process one species: read CSVs, mask sequences, write output."""
    print(f"\n{'='*60}")
    print(f"Processing {species_dir}...")
    print(f"{'='*60}")

    # Parse GFF3. Use robust_listdir because os.listdir flakes with
    # EACCES on some species dirs on this cluster's NFS.
    species_path = os.path.join(RAW_DATA_DIR, species_dir)
    entries = robust_listdir(species_path)
    gff3_files = [f for f in entries if f.endswith(".gff3.gz")]
    gff3_path = os.path.join(RAW_DATA_DIR, species_dir, gff3_files[0])
    genes, gene_list = parse_genes_from_gff3(gff3_path)
    chr_genes = build_chr_gene_index(gene_list)
    print(f"  Loaded {len(genes):,} genes from GFF3")

    # Output directory
    out_dir = os.path.join(OUTPUT_DIR, species_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats = {"total": 0, "masked": 0, "skipped": 0, "masked_bp_list": []}

    for split in ["train", "test", "validation"]:
        csv_in = os.path.join(SEQ2EXP_DIR, species_dir, f"{split}.csv")
        csv_out = os.path.join(out_dir, f"{split}.csv")

        if not os.path.exists(csv_in):
            print(f"  {split}.csv not found, skipping.")
            continue

        rows_out = []
        with open(csv_in, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gene_id = row["name"]
                stats["total"] += 1

                if gene_id not in genes:
                    # Gene not in GFF3; still sanitize assembly-gap N's so
                    # the sequence tokenizes within AgroNT's 1024-token limit.
                    row["sequence"] = sanitize_ns(row["sequence"])
                    rows_out.append(row)
                    stats["skipped"] += 1
                    continue

                masked_seq, n_masked = mask_sequence(
                    row["sequence"], genes[gene_id], chr_genes
                )
                # Also convert any remaining N's (assembly gaps from the source
                # PGB data) to <mask> on 6bp-aligned blocks, so the final
                # sequence tokenizes 1-to-1 regardless of source N content.
                masked_seq = sanitize_ns(masked_seq)
                row["sequence"] = masked_seq
                rows_out.append(row)

                if n_masked > 0:
                    stats["masked"] += 1
                    stats["masked_bp_list"].append(n_masked)

        # Write output CSV
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "sequence", "labels", "split"])
            writer.writeheader()
            writer.writerows(rows_out)

        split_masked = sum(1 for r in rows_out if r["name"] in genes)
        print(f"  {split}: {len(rows_out):,} rows written to {split}.csv")

    # Write metadata
    masked_bp = np.array(stats["masked_bp_list"]) if stats["masked_bp_list"] else np.array([0])
    metadata = {
        "species": species_dir,
        "total_genes": stats["total"],
        "genes_with_masking": stats["masked"],
        "genes_not_in_gff3": stats["skipped"],
        "pct_masked": round(100 * stats["masked"] / max(stats["total"], 1), 1),
        "mean_masked_bp": round(float(np.mean(masked_bp)), 1) if stats["masked"] > 0 else 0,
        "median_masked_bp": round(float(np.median(masked_bp)), 1) if stats["masked"] > 0 else 0,
        "seq_length": 6000,
        "masking_strategy": f"Replace 6bp-aligned blocks overlapping neighboring genes with literal '{MASK_TOKEN}' (tokenizes as 1 special token per 6bp)",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Summary:")
    print(f"    Total genes:        {stats['total']:,}")
    print(f"    Genes masked:       {stats['masked']:,} ({metadata['pct_masked']}%)")
    print(f"    Genes not in GFF3:  {stats['skipped']:,}")
    if stats["masked"] > 0:
        print(f"    Mean masked bp:     {metadata['mean_masked_bp']}")
        print(f"    Median masked bp:   {metadata['median_masked_bp']}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write a README for the dataset
    readme = """# Sequence-to-Expression Data (Masked)

Derived from the [Plant Genomic Benchmark (PGB)](https://huggingface.co/datasets/InstaDeepAI/plant-genomic-benchmark)
seq2exp dataset. Neighboring genes that overlap the 6kb window have been
overwritten with AgroNT's `<mask>` special token (on 6bp-aligned blocks)
so the model sees its own pretraining mask semantics while preserving
one-token-per-6bp tokenization.

## Masking Strategy

For each gene's 6kb window (5kb upstream + 1kb downstream of gene start):
1. All other genes overlapping the window are identified from the Ensembl Plants v56 GFF3.
2. Each neighbor-gene span is snapped outward to the nearest 6bp boundaries,
   then every 6bp block inside the snapped span is overwritten with the
   literal 6-char string `<mask>`. AgroNT's tokenizer emits one `<mask>`
   special token per block, matching the 6bp/token rate of real DNA and
   keeping sequences under the 1024-token position-embedding limit.
3. The target gene's own sequence and promoter region (intergenic space upstream) are preserved.

Snapping to 6bp blocks can slightly over-mask (up to 5bp on each edge)
but guarantees tokenizer alignment.

## Format

Same as the original seq2exp dataset:

| Column | Description |
|--------|-------------|
| `name` | Gene identifier (e.g., AT1G01010) |
| `sequence` | 6000 chars: a mix of DNA and literal `<mask>` 6-char blocks |
| `labels` | JSON-encoded list of log2 expression values (one per tissue) |
| `split` | `train`, `test`, or `validation` |
"""
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
        f.write(readme)

    for species in SPECIES:
        process_species(species)

    print("\nDone! Masked dataset written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
