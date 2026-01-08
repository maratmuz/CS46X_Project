"""
Utility to list sequences in a FASTA that meet a minimum length.
Helps pick sequences long enough for the context+prediction window (e.g., >= 1,001,000 bp).
"""

import argparse
from pathlib import Path

from Bio import SeqIO


def list_sequences(fasta_path: Path, min_len: int):
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq_len = len(record.seq)
        if seq_len >= min_len:
            print(f"{record.id}\t{seq_len}")


def main():
    parser = argparse.ArgumentParser(description="List sequences meeting a minimum length in a FASTA.")
    parser.add_argument("fasta", type=Path, help="Path to FASTA file.")
    parser.add_argument(
        "--min-len",
        type=int,
        default=1_001_000,
        help="Minimum length required (default: 1,001,000).",
    )
    args = parser.parse_args()

    if not args.fasta.exists():
        raise SystemExit(f"FASTA not found: {args.fasta}")

    list_sequences(args.fasta, args.min_len)


if __name__ == "__main__":
    main()
