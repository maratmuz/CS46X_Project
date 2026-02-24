# Marat/filter_long_genes.py

"""
Filter CDS annotations to keep only long genes (cds_length > min-length).

Usage from repo root:
  python scripts/data/filter_long_genes.py --input scripts/data/output/tair10_cds_annotations.csv --min-length 8000
"""

from pathlib import Path
import pandas as pd
import argparse

DEFAULT_INPUT = Path("scripts/data/output/tair10_cds_annotations.csv")
DEFAULT_OUTPUT = Path("scripts/data/output/tair10_cds_annotations_len_gt_8000.csv")
DEFAULT_MIN_LEN = 8000


def main():
    parser = argparse.ArgumentParser(description="Filter CDS annotations to retain long genes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV produced by gff_fasta_to_csv.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV for filtered genes.",
    )
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LEN, help="Minimum CDS length to keep.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    print(f"Loading input CSV from {args.input} ...")
    df = pd.read_csv(args.input)

    if "cds_length" not in df.columns:
        raise KeyError(
            "Expected a 'cds_length' column in the input CSV but did not find one."
        )

    # Ensure cds_length is numeric
    df["cds_length"] = pd.to_numeric(df["cds_length"], errors="coerce")

    before = len(df)
    df_long = df[df["cds_length"] > args.min_length].copy()
    after = len(df_long)

    print(f"Total rows in input: {before}")
    print(f"Rows with cds_length > {args.min_length}: {after}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(args.output, index=False)
    print(f"Wrote filtered CSV to {args.output}")


if __name__ == "__main__":
    main()
