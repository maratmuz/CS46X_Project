# Marat/filter_long_genes.py

"""
Filter CDS annotations to keep only long genes (cds_length > 8000).

Assumes the input CSV was produced by gff_fasta_to_csv.py and has columns:
  seqid, gene_id, gene_name, transcript_id, strand,
  cds_start, cds_end, cds_length, n_cds_exons,
  transcript_attributes, cds_sequence

Usage from CS46X_Project root:

  python Marat/filter_long_genes.py
"""

from pathlib import Path
import pandas as pd

# Hard-coded paths (relative to repo root)
INPUT_PATH = Path("Marat/output/tair10_cds_annotations.csv")
OUTPUT_PATH = Path("Marat/output/tair10_cds_annotations_len_gt_8000.csv")

MIN_LENGTH = 8000  # nucleotides


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_PATH}")

    print(f"Loading input CSV from {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)

    if "cds_length" not in df.columns:
        raise KeyError(
            "Expected a 'cds_length' column in the input CSV but did not find one."
        )

    # Ensure cds_length is numeric
    df["cds_length"] = pd.to_numeric(df["cds_length"], errors="coerce")

    before = len(df)
    df_long = df[df["cds_length"] > MIN_LENGTH].copy()
    after = len(df_long)

    print(f"Total rows in input: {before}")
    print(f"Rows with cds_length > {MIN_LENGTH}: {after}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote filtered CSV to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
