from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence, Tuple

from Bio import SeqIO


class GenomicDataLoader:
    """
    Handles FASTA loading and provides multiple read modes:
      - random: random sample, random start
      - unique_start: cycle through unique samples from start
      - midpoint: prompt is first 50% of sequence, target is next pred_len
      - offset: contiguous slice from a concatenated sequence by offset
    """

    def __init__(self):
        self.supported_formats = {"fasta"}
        self._data: list = []
        self._selected_samples: list | None = None
        self._sample_index = 0
        self._full_sequence: str | None = None

    def load(self, path: str | Path, format: str, verbose: bool = False):
        path = Path(path)
        fmt = format.lower()
        if fmt not in self.supported_formats:
            raise ValueError(f'Format "{format}" was not found in the list of supported data types.')

        if fmt == "fasta":
            self._data = list(SeqIO.parse(path, "fasta"))
            if not self._data:
                raise ValueError(f"No sequences found in {path}")
            self._full_sequence = "".join(str(record.seq) for record in self._data)

            if verbose:
                print(f"\nLoaded {len(self._data)} sequences from {path}")
                print("=" * 70)
                for idx, seq in enumerate(self._data, 1):
                    seq_len = len(seq)
                    mid_point = 2 * (seq_len // 4)
                    print(f"  Sample {idx}: {seq.id[:40]:40} | Length: {seq_len:8,} | Midpoint: {mid_point:8,}")
                print("=" * 70)
                avg_len = sum(len(s) for s in self._data) / len(self._data)
                print(f"Average sequence length: {avg_len:,.0f}\n")

    def _ensure_loaded(self):
        if not self._data:
            raise RuntimeError("Data not loaded. Call load() first.")

    def read_random(self, splits: Sequence[int]) -> list[str]:
        self._ensure_loaded()
        sampled_seq = random.choice(self._data)
        seq_len = len(sampled_seq)
        req_len = sum(splits)
        if req_len > seq_len:
            raise ValueError(f"Required length {req_len} exceeds sequence length {seq_len}")

        seq_start = random.randint(0, seq_len - req_len)
        data: list[str] = []

        for split in splits:
            split_data = sampled_seq[seq_start : seq_start + split]
            seq_start += split
            data.append(str(split_data.seq))

        return data

    def initialize_unique_samples(self, num_samples: int = 4):
        self._ensure_loaded()
        if num_samples > len(self._data):
            raise ValueError(f"Cannot select {num_samples} samples from {len(self._data)} available sequences")

        self._selected_samples = random.sample(self._data, num_samples)
        self._sample_index = 0

    def read_unique_start(self, splits: Sequence[int]) -> list[str]:
        if self._selected_samples is None:
            raise RuntimeError("Must call initialize_unique_samples() before using read_unique_start()")

        sampled_seq = self._selected_samples[self._sample_index % len(self._selected_samples)]
        self._sample_index += 1

        seq_len = len(sampled_seq)
        req_len = sum(splits)
        if req_len > seq_len:
            raise ValueError(f"Required length {req_len} exceeds sequence length {seq_len}")

        seq_start = 0
        data: list[str] = []
        for split in splits:
            split_data = sampled_seq[seq_start : seq_start + split]
            seq_start += split
            data.append(str(split_data.seq))

        return data

    def read_midpoint(self, pred_len: int) -> Tuple[str, str]:
        if self._selected_samples is None:
            raise RuntimeError("Must call initialize_unique_samples() before using read_midpoint()")

        sampled_seq = self._selected_samples[self._sample_index % len(self._selected_samples)]
        self._sample_index += 1

        seq_len = len(sampled_seq)
        mid_point = 2 * (seq_len // 4)

        if mid_point + pred_len > seq_len:
            raise ValueError(f"Sequence length {seq_len} is too short for midpoint {mid_point} + pred_len {pred_len}")

        prompt = str(sampled_seq[:mid_point].seq)
        target = str(sampled_seq[mid_point : mid_point + pred_len].seq)
        return prompt, target

    def read_offset(self, seq_len: int, pred_len: int, offset: int = 0) -> Tuple[str, str]:
        """
        Deterministic contiguous slice from the concatenated sequence using an offset.
        """
        if self._full_sequence is None:
            raise RuntimeError("Full sequence cache not available; ensure load() was called.")

        req_len = seq_len + pred_len
        if req_len + offset > len(self._full_sequence):
            raise ValueError(
                f"Requested offset {offset} with length {req_len} exceeds available length {len(self._full_sequence)}"
            )

        start = offset
        prompt = self._full_sequence[start : start + seq_len]
        target = self._full_sequence[start + seq_len : start + req_len]
        if len(prompt) != seq_len or len(target) != pred_len:
            raise ValueError("Failed to read the requested lengths; check sequence boundaries.")
        return prompt, target


def read_sequence(file_path: str | Path) -> str:
    sequence = []
    skip_next = False
    with Path(file_path).open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                skip_next = True
                continue
            if skip_next:
                skip_next = False
                continue
            sequence.append(line)
    return "".join(sequence)


def read_chars(num_chars: int, file_path: str | Path, offset: int = 0) -> str:
    full_seq = read_sequence(file_path)
    end = offset + num_chars
    if end > len(full_seq):
        raise ValueError(f"Requested {num_chars} chars from offset {offset}, but sequence length is {len(full_seq)}")
    return full_seq[offset:end]
