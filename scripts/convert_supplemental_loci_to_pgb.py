#!/usr/bin/env python3
"""Convert build_supplemental_loci outputs into extractor-compatible parquet.

The frozen-embedding pipeline in ``shared/plant-seq-2-expr`` expects this
layout::

    <dataset>/<species>/{train,validation,test}.parquet

and each parquet must contain ``sequence``, ``name``, and ``labels``. This
converter combines each species' nuclear pseudogene and intergenic FASTA/TSV
pairs, preserves their metadata, and assigns a one-element binary label:

    intergenic_locus     -> [0.0]
    annotated_pseudogene -> [1.0]

The splits are deterministic and stratified by locus category. They organise
rows for the extractor; embedding extraction itself does not depend on which
split a sequence occupies. A scientific classifier benchmark may require a
stricter chromosome- or homology-aware split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_VERSION = "1.0"
SPECIES = (
    "arabidopsis_thaliana",
    "glycine_max",
    "oryza_sativa",
    "solanum_lycopersicum",
    "zea_mays",
)
SPLITS = ("train", "validation", "test")
SOURCE_FILES = (
    ("nuclear_intergenic", "intergenic_locus", 0.0),
    ("nuclear_pseudogenes", "annotated_pseudogene", 1.0),
)
REQUIRED_COLUMNS = ("sequence", "name", "labels")
DATASET_VARIANTS = {
    "intergenic": "intergenic_locus",
    "pseudogenes": "annotated_pseudogene",
}
SUPPORTED_BASES = set("ACGTN")
IUPAC_AMBIGUOUS_BASES = set("RYSWKMBDHV")
IUPAC_TO_N = str.maketrans({base: "N" for base in IUPAC_AMBIGUOUS_BASES})


class ConversionError(RuntimeError):
    """Raised when source data cannot safely be converted."""


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(dataset_id, sequence)`` from a wrapped FASTA file."""
    header: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                full_header = line[1:]
                header = full_header.split("|", 1)[0].split(None, 1)[0]
                if not header:
                    raise ConversionError(f"{path}:{line_number}: empty FASTA identifier")
                chunks = []
            else:
                if header is None:
                    raise ConversionError(
                        f"{path}:{line_number}: sequence appears before the first FASTA header"
                    )
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks).upper()


def load_sequences(path: Path, expected_length: int) -> tuple[dict[str, str], dict[str, int]]:
    sequences: dict[str, str] = {}
    normalized_counts: dict[str, int] = {}
    for dataset_id, sequence in iter_fasta(path):
        if dataset_id in sequences:
            raise ConversionError(f"{path}: duplicate FASTA identifier {dataset_id!r}")
        if len(sequence) != expected_length:
            raise ConversionError(
                f"{path}: {dataset_id!r} has length {len(sequence)}, expected {expected_length}"
            )
        invalid = set(sequence) - SUPPORTED_BASES - IUPAC_AMBIGUOUS_BASES
        if invalid:
            raise ConversionError(
                f"{path}: {dataset_id!r} contains unsupported bases {sorted(invalid)!r}"
            )
        normalized_counts[dataset_id] = sum(
            sequence.count(base) for base in IUPAC_AMBIGUOUS_BASES
        )
        sequences[dataset_id] = sequence.translate(IUPAC_TO_N)
    return sequences, normalized_counts


def load_source(
    species_dir: Path,
    basename: str,
    expected_category: str,
    label: float,
    expected_length: int,
) -> pd.DataFrame:
    fasta_path = species_dir / f"{basename}.fasta"
    metadata_path = species_dir / f"{basename}.tsv"
    for path in (fasta_path, metadata_path):
        if not path.is_file():
            raise ConversionError(f"Required source file is missing: {path}")

    sequences, normalized_counts = load_sequences(fasta_path, expected_length)
    metadata = pd.read_csv(metadata_path, sep="\t", keep_default_na=False)
    required_metadata = {"dataset_id", "locus_category", "species"}
    missing = required_metadata - set(metadata.columns)
    if missing:
        raise ConversionError(
            f"{metadata_path}: missing required columns {sorted(missing)!r}"
        )
    if metadata["dataset_id"].duplicated().any():
        duplicate = metadata.loc[metadata["dataset_id"].duplicated(), "dataset_id"].iloc[0]
        raise ConversionError(f"{metadata_path}: duplicate dataset_id {duplicate!r}")

    metadata_ids = set(metadata["dataset_id"].astype(str))
    fasta_ids = set(sequences)
    if metadata_ids != fasta_ids:
        only_tsv = sorted(metadata_ids - fasta_ids)[:5]
        only_fasta = sorted(fasta_ids - metadata_ids)[:5]
        raise ConversionError(
            f"{basename}: FASTA/TSV identifier mismatch; "
            f"TSV-only examples={only_tsv!r}, FASTA-only examples={only_fasta!r}"
        )

    categories = set(metadata["locus_category"].astype(str))
    if categories != {expected_category}:
        raise ConversionError(
            f"{metadata_path}: expected category {expected_category!r}, found {sorted(categories)!r}"
        )

    metadata.insert(0, "sequence", metadata["dataset_id"].map(sequences))
    metadata.insert(1, "name", metadata["dataset_id"].astype(str))
    metadata.insert(2, "labels", [[label] for _ in range(len(metadata))])
    metadata.insert(3, "binary_label", int(label))
    metadata.insert(4, "label_name", expected_category)
    metadata.insert(5, "source_fasta", fasta_path.name)
    metadata.insert(
        6,
        "normalized_ambiguous_bases",
        metadata["dataset_id"].map(normalized_counts).astype(int),
    )
    return metadata


def allocated_counts(total: int, fractions: Sequence[float]) -> list[int]:
    """Use largest remainders so the per-class split counts sum to ``total``."""
    raw = [total * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stratified_split(
    frame: pd.DataFrame,
    species: str,
    fractions: Sequence[float],
    seed: int,
) -> dict[str, pd.DataFrame]:
    split_indices: dict[str, list[int]] = {split: [] for split in SPLITS}
    for label_name, group in frame.groupby("label_name", sort=True):
        indices = group.index.tolist()
        random.Random(stable_seed(seed, species, str(label_name))).shuffle(indices)
        counts = allocated_counts(len(indices), fractions)
        offset = 0
        for split, count in zip(SPLITS, counts):
            split_indices[split].extend(indices[offset : offset + count])
            offset += count

    result: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        indices = split_indices[split]
        random.Random(stable_seed(seed, species, split)).shuffle(indices)
        result[split] = frame.loc[indices].reset_index(drop=True)
    return result


def validate_output(frame: pd.DataFrame, expected_length: int) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ConversionError(f"Converted frame is missing columns {sorted(missing)!r}")
    if frame["name"].duplicated().any():
        raise ConversionError("Converted dataset contains duplicate names")
    bad_lengths = frame["sequence"].str.len() != expected_length
    if bad_lengths.any():
        raise ConversionError("Converted dataset contains a sequence with an invalid length")
    if not frame["labels"].map(lambda value: len(value) == 1).all():
        raise ConversionError("Every labels value must be a one-element vector")


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        # Explicitly disable Arrow's column-conversion thread pool. Some HPC
        # login nodes impose a low process/thread limit, while this conversion
        # is small enough that parallel column conversion is unnecessary.
        table = pa.Table.from_pandas(frame, preserve_index=False, nthreads=1)
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in sorted(Counter(frame["label_name"]).items())
    }


def convert_species(
    input_root: Path,
    output_root: Path,
    species: str,
    fractions: Sequence[float],
    seed: int,
    expected_length: int,
) -> dict[str, object]:
    species_dir = input_root / species
    frames = [
        load_source(species_dir, basename, category, label, expected_length)
        for basename, category, label in SOURCE_FILES
    ]
    complete = pd.concat(frames, ignore_index=True, sort=False)
    validate_output(complete, expected_length)
    split_frames = stratified_split(complete, species, fractions, seed)

    output_species = output_root / species
    output_species.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "total": len(complete),
        "classes": class_counts(complete),
        "rows_with_normalized_ambiguous_bases": int(
            (complete["normalized_ambiguous_bases"] > 0).sum()
        ),
        "normalized_ambiguous_base_count": int(
            complete["normalized_ambiguous_bases"].sum()
        ),
        "splits": {},
        "separate_datasets": {
            variant: {"total": 0, "classes": {}, "splits": {}}
            for variant in DATASET_VARIANTS
        },
    }
    all_names: set[str] = set()
    for split, frame in split_frames.items():
        validate_output(frame, expected_length)
        overlap = all_names.intersection(frame["name"])
        if overlap:
            raise ConversionError(f"{species}: rows overlap between splits: {sorted(overlap)[:5]!r}")
        all_names.update(frame["name"])
        path = output_species / f"{split}.parquet"
        write_parquet_atomic(frame, path)
        summary["splits"][split] = {
            "rows": len(frame),
            "classes": class_counts(frame),
            "path": str(path),
        }
        for variant, label_name in DATASET_VARIANTS.items():
            subset = frame.loc[frame["label_name"] == label_name].reset_index(drop=True)
            subset_dir = output_root / variant / species
            subset_dir.mkdir(parents=True, exist_ok=True)
            subset_path = subset_dir / f"{split}.parquet"
            write_parquet_atomic(subset, subset_path)
            variant_summary = summary["separate_datasets"][variant]
            variant_summary["total"] += len(subset)
            variant_summary["classes"] = class_counts(subset)
            variant_summary["splits"][split] = {
                "rows": len(subset),
                "classes": class_counts(subset),
                "path": str(subset_path),
            }
    if len(all_names) != len(complete):
        raise ConversionError(f"{species}: split union does not reproduce every input row")
    return summary


def parse_fractions(values: Iterable[float]) -> tuple[float, float, float]:
    fractions = tuple(values)
    if len(fractions) != 3:
        raise argparse.ArgumentTypeError("exactly three split fractions are required")
    if any(value <= 0 for value in fractions):
        raise argparse.ArgumentTypeError("split fractions must be positive")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise argparse.ArgumentTypeError("split fractions must sum to 1")
    return fractions  # type: ignore[return-value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="supplemental_loci output root")
    parser.add_argument("--output", type=Path, required=True, help="PGB-compatible parquet root")
    parser.add_argument(
        "--split-fractions",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VALIDATION", "TEST"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-length", type=int, default=6000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the converter's parquet/manifest files in an existing output directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    try:
        fractions = parse_fractions(args.split_fractions)
        if not input_root.is_dir():
            raise ConversionError(f"Input directory does not exist: {input_root}")
        if output_root.exists() and any(output_root.iterdir()) and not args.force:
            raise ConversionError(
                f"Output directory is nonempty: {output_root}. Use a new path or pass --force."
            )
        output_root.mkdir(parents=True, exist_ok=True)

        species_summary = {
            species: convert_species(
                input_root,
                output_root,
                species,
                fractions,
                args.seed,
                args.expected_length,
            )
            for species in SPECIES
        }
        manifest = {
            "format": "plant-seq-2-expr-pgb-compatible-v1",
            "script_version": SCRIPT_VERSION,
            "source": str(input_root),
            "output": str(output_root),
            "seed": args.seed,
            "split_fractions": dict(zip(SPLITS, fractions)),
            "sequence_length": args.expected_length,
            "required_columns": list(REQUIRED_COLUMNS),
            "label_mapping": {
                "0": "intergenic_locus",
                "1": "annotated_pseudogene",
            },
            "species": species_summary,
        }
        for variant, label_name in DATASET_VARIANTS.items():
            variant_root = output_root / variant
            variant_manifest = {
                "format": "plant-seq-2-expr-pgb-compatible-v1",
                "script_version": SCRIPT_VERSION,
                "dataset_variant": variant,
                "locus_category": label_name,
                "source": str(input_root),
                "output": str(variant_root),
                "seed": args.seed,
                "split_fractions": dict(zip(SPLITS, fractions)),
                "sequence_length": args.expected_length,
                "required_columns": list(REQUIRED_COLUMNS),
                "label_mapping": manifest["label_mapping"],
                "species": {
                    species: summary["separate_datasets"][variant]
                    for species, summary in species_summary.items()
                },
            }
            variant_manifest_path = variant_root / "manifest.json"
            variant_temporary = variant_manifest_path.with_name(
                f".{variant_manifest_path.name}.tmp-{os.getpid()}"
            )
            try:
                variant_temporary.write_text(
                    json.dumps(variant_manifest, indent=2, sort_keys=True) + "\n"
                )
                os.replace(variant_temporary, variant_manifest_path)
            finally:
                if variant_temporary.exists():
                    variant_temporary.unlink()
        manifest_path = output_root / "manifest.json"
        temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()

        print(f"Wrote extractor-compatible dataset to {output_root}")
        for species, summary in species_summary.items():
            split_text = ", ".join(
                f"{split}={details['rows']}" for split, details in summary["splits"].items()
            )
            print(f"  {species}: total={summary['total']} ({split_text})")
        print(f"Manifest: {manifest_path}")
        return 0
    except (ConversionError, OSError, ValueError, ImportError, argparse.ArgumentTypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
