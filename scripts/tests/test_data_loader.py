"""
Helpers to load chromosome and CDS windows with low N content for testing.

We favor windows of a specific length (e.g., 33,000 = 32k context + 1k pred)
and discard windows with excessive unknown bases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

from Bio import SeqIO

Window = Tuple[str, int, str]  # (record_id, start, sequence)


def _iter_windows(seq: str, record_id: str, window_len: int, stride: int) -> Iterator[Window]:
    for start in range(0, len(seq) - window_len + 1, stride):
        yield record_id, start, seq[start : start + window_len]


def _n_fraction(seq: str) -> float:
    if not seq:
        return 1.0
    n_count = seq.upper().count("N")
    return n_count / len(seq)


def get_chrom_windows(
    fasta_path: Path,
    window_len: int,
    max_windows: int,
    max_n_fraction: float = 0.02,
    stride: int | None = None,
) -> List[Window]:
    """
    Extract windows from chromosome FASTA records with limited N content.
    """
    stride = stride or window_len
    windows: List[Window] = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq_str = str(record.seq)
        for rec_id, start, window in _iter_windows(seq_str, record.id, window_len, stride):
            if _n_fraction(window) <= max_n_fraction:
                windows.append((rec_id, start, window))
            if len(windows) >= max_windows:
                return windows
    return windows


def get_cds_windows(
    fasta_path: Path,
    window_len: int,
    max_windows: int,
    max_n_fraction: float = 0.02,
) -> List[Window]:
    """
    Extract windows from CDS FASTA (longest CDS). Skips entries shorter than window_len.
    """
    windows: List[Window] = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq_str = str(record.seq)
        if len(seq_str) < window_len:
            continue
        window = seq_str[:window_len]
        if _n_fraction(window) <= max_n_fraction:
            windows.append((record.id, 0, window))
        if len(windows) >= max_windows:
            break
    return windows
