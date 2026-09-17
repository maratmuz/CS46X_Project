#!/usr/bin/env python3
"""Create a row-aligned, three-class view of the raw-TPM PGB dataset.

The source and its train/validation/test rows are never modified or shuffled.
Each output parquet retains every input column and adds vector-valued TPM class
targets aligned with the existing multi-tissue ``labels`` vector.  Embeddings
remain in the frozen-embedding cache; the manifest records their paths and the
``embedding_row`` column records the exact row used from each cached tensor.

Examples:
  python scripts/make_tpm_classification_dataset.py build
  python scripts/make_tpm_classification_dataset.py inspect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/nfs/hpc/share/evo2_shared"))
DEFAULT_INPUT = SHARED_DIR / "datasets" / "pgb_parquet_tpm"
DEFAULT_LOG2_INPUT = SHARED_DIR / "datasets" / "pgb_parquet_tpm_log2"
DEFAULT_REFERENCE = SHARED_DIR / "datasets" / "pgb_parquet"
DEFAULT_OUTPUT = SHARED_DIR / "datasets" / "pgb_parquet_tpm_3class"
DEFAULT_EMBED_ROOT = SHARED_DIR / "frozen-embeddings"

SPECIES = (
    "arabidopsis_thaliana",
    "glycine_max",
    "oryza_sativa",
    "solanum_lycopersicum",
    "zea_mays",
)
SPLITS = ("train", "validation", "test")
CLASS_NAMES = ("low", "medium", "high")
EMBEDDING_MODELS = ("agront", "evo2", "plantcad2", "ntv3")
INVALID_CLASS = -1
INVALID_CLASS_NAME = "missing_or_invalid"

# Target provenance from AgroNT Supplementary Data 2. Arabidopsis and soybean
# retain one target per listed run (matching their 56- and 14-value PGB label
# vectors); the other species aggregate runs by tissue in first-listed order.
ARABIDOPSIS_TARGETS = (
    ("carpel", "ERR3333421"), ("cauline leaf", "ERR3333391"),
    ("cotyledon", "ERR3333406"), ("cotyledon", "ERR3333412"),
    ("egg-cell like callus", "ERR3333402"), ("flower", "ERR3333398"),
    ("flower", "ERR3333426"), ("flower", "ERR3333427"),
    ("flower", "ERR3333428"), ("flower", "ERR3333429"),
    ("flower", "ERR3333430"), ("flower pedicel", "ERR3333397"),
    ("hypocotyl", "ERR3333407"), ("petal", "ERR3333399"),
    ("plant callus", "ERR3333403"), ("pollen", "ERR3333443"),
    ("root", "ERR3333396"), ("root cell culture", "ERR3333404"),
    ("root cell culture", "ERR3333405"), ("root tip", "ERR3333408"),
    ("root upper zone", "ERR3333409"), ("rosette leaf 1", "ERR3333413"),
    ("rosette leaf 10", "ERR3333423"), ("rosette leaf 11", "ERR3333424"),
    ("rosette leaf 12", "ERR3333425"), ("rosette leaf 2", "ERR3333414"),
    ("rosette leaf 3", "ERR3333415"), ("rosette leaf 4", "ERR3333416"),
    ("rosette leaf 5", "ERR3333417"), ("rosette leaf 6", "ERR3333418"),
    ("rosette leaf 7", "ERR3333419"),
    ("rosette leaf 7, distal part", "ERR3333392"),
    ("rosette leaf 7, petiole", "ERR3333394"),
    ("rosette leaf 7, proximal part", "ERR3333393"),
    ("rosette leaf 8", "ERR3333420"), ("rosette leaf 9", "ERR3333422"),
    ("seed", "ERR3333436"), ("seed", "ERR3333437"),
    ("seed", "ERR3333438"), ("seed", "ERR3333439"),
    ("seed", "ERR3333440"), ("seed", "ERR3333441"),
    ("seed", "ERR3333442"), ("senescent leaf", "ERR3333395"),
    ("sepal", "ERR3333388"),
    ("shoot apical meristem, cotyledons and first leaves", "ERR3333411"),
    ("silique", "ERR3333431"), ("silique", "ERR3333432"),
    ("silique", "ERR3333433"), ("silique", "ERR3333434"),
    ("silique", "ERR3333435"), ("silique septum", "ERR3333400"),
    ("silique valves", "ERR3333401"), ("stamen", "ERR3333410"),
    ("stem, 1st node", "ERR3333389"), ("stem, 2nd internode", "ERR3333390"),
)

SOYBEAN_TARGETS = (
    ("12HA1 IN RH", "SRR037374"), ("12HA1 IN RH", "SRR037375"),
    ("24HA1 IN RH", "SRR037376"), ("24HA1 IN RH", "SRR037377"),
    ("48HA1 IN RH", "SRR037378"), ("48HA1 IN RH", "SRR037379"),
    ("48HA1 IN RH", "SRR037380"), ("Apical Meristem", "SRR037381"),
    ("Flower Pods", "SRR037382"), ("Green Pods", "SRR037383"),
    ("Leaves", "SRR037384"), ("Nodule", "SRR037385"),
    ("Root", "SRR037387"), ("Root", "SRR037386"),
)

AGGREGATED_TARGETS = {
    "zea_mays": (
        ("endosperm 12 days after pollination", ("SRR957415", "SRR957416", "SRR957417")),
        ("pericarp and aleurone", ("SRR957418", "SRR957419", "SRR957420")),
        ("endosperm crown", ("SRR957421", "SRR957422", "SRR957423")),
        ("symmetrical division zone", ("SRR957424", "SRR957425", "SRR957426")),
        ("stomatal division zone", ("SRR957427", "SRR957428", "SRR957429")),
        ("growth zone", ("SRR957430", "SRR957431", "SRR957432")),
        ("embryos 20 days after pollination", ("SRR957433", "SRR957434", "SRR957435")),
        ("embryos", ("SRR957436", "SRR957437", "SRR957438")),
        ("germinating kernels", ("SRR957439", "SRR957440", "SRR957441")),
        ("mature leaf tissue (leaf 8)", ("SRR957442", "SRR957443", "SRR957444")),
        ("6-8 mm from tip of ear primordium", ("SRR957445", "SRR957446", "SRR957447")),
        ("2-4 mm from tip of ear primordium", ("SRR957448", "SRR957449", "SRR957450")),
        ("Root maturation zone", ("SRR957451", "SRR957454", "SRR957457")),
        ("Root elongation zone", ("SRR957452", "SRR957455", "SRR957458")),
        ("Root cortex", ("SRR957453", "SRR957456", "SRR957459")),
        ("Primary root", ("SRR957460", "SRR957461", "SRR957462")),
        ("Secondary root", ("SRR957463", "SRR957464", "SRR957465")),
        ("Mature pollen", ("SRR957466", "SRR957467", "SRR957468")),
        ("silks", ("SRR957469", "SRR957470", "SRR957471")),
        ("mature female spikelets", ("SRR957472", "SRR957473", "SRR957474")),
        ("Internode 6-7", ("SRR957475", "SRR957476", "SRR957477")),
        ("Internode 7-8", ("SRR957478", "SRR957479", "SRR957480")),
        ("Vegetative Meristem Surrounding Tissue", ("SRR957481", "SRR957482")),
    ),
    "oryza_sativa": (
        ("anther wall", ("ERR6907776", "ERR6907777", "ERR6907778")),
        ("leaf collar", ("ERR6907779", "ERR6907780", "ERR6907781", "ERR6907782")),
        ("leaf sheath", ("ERR6907788", "ERR6907789")),
        ("ligule", ("ERR6907783", "ERR6907784")),
        ("root hair", ("ERR6907785", "ERR6907786", "ERR6907787")),
        ("anther", ("ERR3326983", "ERR3326984", "ERR3326985")),
        ("pollen", ("ERR3326986", "ERR3326987", "ERR3326988")),
    ),
    "solanum_lycopersicum": (
        ("leaf", ("SRR404309", "SRR404310")),
        ("root", ("SRR404311", "SRR404312")),
        ("flower", ("SRR404313", "SRR404314")),
        ("flower bud", ("SRR404315", "SRR404316")),
        ("1cm fruit", ("SRR404317", "SRR404318")),
        ("2cm fruit", ("SRR404319", "SRR404320")),
        ("3cm fruit", ("SRR404321", "SRR404322")),
        ("mature green fruit", ("SRR404324", "SRR404325")),
        ("breaker fruit", ("SRR404326", "SRR404327")),
        ("fruit at 10 days after the breaker stage", ("SRR404328", "SRR404329")),
    ),
}

SOURCE_DATASETS = {
    "arabidopsis_thaliana": "E-MTAB-7978",
    "glycine_max": "SRA012188",
    "oryza_sativa": "PRJEB47919 and PRJEB32629",
    "solanum_lycopersicum": "E-MTAB-4812",
    "zea_mays": "GSE50191",
}


class DatasetError(RuntimeError):
    pass


def tissue_targets(species: str, n_targets: int) -> list[dict[str, object]]:
    if species == "arabidopsis_thaliana":
        rows = ARABIDOPSIS_TARGETS
        result = [
            {"target_index": i, "target_name": f"{tissue} [{run}]", "tissue": tissue,
             "source_run_ids": [run], "aggregation": "none; one PGB target per run"}
            for i, (tissue, run) in enumerate(rows)
        ]
    elif species == "glycine_max":
        rows = SOYBEAN_TARGETS
        result = [
            {"target_index": i, "target_name": f"{tissue} [{run}]", "tissue": tissue,
             "source_run_ids": [run], "aggregation": "none; one PGB target per run"}
            for i, (tissue, run) in enumerate(rows)
        ]
    else:
        result = [
            {"target_index": i, "target_name": tissue, "tissue": tissue,
             "source_run_ids": list(runs), "aggregation": "mean across listed runs"}
            for i, (tissue, runs) in enumerate(AGGREGATED_TARGETS[species])
        ]
    if len(result) != n_targets:
        raise DatasetError(
            f"{species}: PGB has {n_targets} targets but tissue metadata has {len(result)}"
        )
    return result


def stack_vectors(series: pd.Series, description: str) -> np.ndarray:
    try:
        rows = [np.asarray(value, dtype=np.float64) for value in series]
        if not rows:
            raise DatasetError(f"{description}: split is empty")
        widths = {row.shape for row in rows}
        if len(widths) != 1 or rows[0].ndim != 1:
            raise DatasetError(f"{description}: labels are not uniform one-dimensional vectors")
        return np.stack(rows)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{description}: labels are not numeric vectors: {error}") from error


def assign_classes(
    tpm: np.ndarray, low_threshold: float, high_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return integer classes, readable names, and a valid-value mask."""
    if not math.isfinite(low_threshold) or not math.isfinite(high_threshold):
        raise DatasetError("thresholds must be finite")
    if low_threshold < 0 or high_threshold <= low_threshold:
        raise DatasetError("require 0 <= low-threshold < high-threshold")
    valid = np.isfinite(tpm) & (tpm >= 0)
    classes = np.full(tpm.shape, INVALID_CLASS, dtype=np.int8)
    classes[valid & (tpm < low_threshold)] = 0
    classes[valid & (tpm >= low_threshold) & (tpm <= high_threshold)] = 1
    classes[valid & (tpm > high_threshold)] = 2
    if np.any(valid & ((classes < 0) | (classes > 2))):
        raise DatasetError("at least one usable TPM value did not receive exactly one class")
    if np.any((~valid) & (classes != INVALID_CLASS)):
        raise DatasetError("an invalid TPM value received a usable class")
    names = np.full(tpm.shape, INVALID_CLASS_NAME, dtype=object)
    for index, name in enumerate(CLASS_NAMES):
        names[classes == index] = name
    return classes, names, valid


def names_digest(names: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in names:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False, nthreads=1)
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_reference_alignment(source: pd.DataFrame, reference_path: Path, description: str) -> None:
    if not reference_path.is_file():
        raise DatasetError(f"reference PGB split is missing: {reference_path}")
    reference = pd.read_parquet(reference_path, columns=["name", "sequence"])
    if len(reference) != len(source):
        raise DatasetError(f"{description}: source/reference row count differs")
    if not np.array_equal(reference["name"].astype(str).to_numpy(), source["name"].astype(str).to_numpy()):
        raise DatasetError(f"{description}: source/reference gene IDs or row order differ")
    if not np.array_equal(reference["sequence"].to_numpy(), source["sequence"].to_numpy()):
        raise DatasetError(f"{description}: source/reference sequences or row order differ")


def discover_embedding_references(embed_root: Path) -> dict[str, object]:
    found: dict[str, object] = OrderedDict()
    for model in EMBEDDING_MODELS:
        model_root = embed_root / model
        if not model_root.is_dir():
            continue
        sizes: dict[str, object] = OrderedDict()
        for size_root in sorted(path for path in model_root.iterdir() if path.is_dir()):
            files = [
                str(path) for path in sorted(size_root.glob("*/*.pt"))
                if path.stem.split("-", 1)[0] in SPLITS
            ]
            if files:
                sizes[size_root.name] = {"files": files}
        if sizes:
            found[model] = sizes
    return found


def distribution_rows(
    species: str,
    split: str,
    classes: np.ndarray,
    valid: np.ndarray,
    targets: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for index, target in enumerate(targets):
        usable = int(valid[:, index].sum())
        total = len(classes)
        row: dict[str, object] = {
            "species": species,
            "tissue_index": index,
            "tissue": target["tissue"],
            "target_name": target["target_name"],
            "source_run_ids": ";".join(target["source_run_ids"]),
            "split": split,
            "total": total,
            "usable": usable,
            "missing_invalid": total - usable,
        }
        for class_index, class_name in enumerate(CLASS_NAMES):
            count = int((classes[:, index] == class_index).sum())
            row[f"{class_name}_count"] = count
            row[f"{class_name}_percent"] = 100.0 * count / usable if usable else np.nan
        rows.append(row)
    return rows


def aggregate_distribution(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_columns = ["total", "usable", "missing_invalid"] + [f"{name}_count" for name in CLASS_NAMES]
    tissue = detail.groupby(
        ["species", "tissue_index", "tissue", "target_name", "source_run_ids"],
        as_index=False, sort=False,
    )[count_columns].sum()
    for name in CLASS_NAMES:
        tissue[f"{name}_percent"] = np.where(
            tissue["usable"] > 0, 100.0 * tissue[f"{name}_count"] / tissue["usable"], np.nan
        )

    records = []
    for species, group in detail.groupby("species", sort=False):
        record = {"species": species}
        for column in count_columns:
            record[column] = int(group[column].sum())
        records.append(record)
    all_record = {"species": "ALL_SPECIES"}
    for column in count_columns:
        all_record[column] = int(detail[column].sum())
    records.append(all_record)
    overall = pd.DataFrame(records)
    for name in CLASS_NAMES:
        overall[f"{name}_percent"] = np.where(
            overall["usable"] > 0,
            100.0 * overall[f"{name}_count"] / overall["usable"],
            np.nan,
        )
    return tissue, overall


def warning_messages(detail: pd.DataFrame, small_class_min: int) -> list[str]:
    warnings = []
    for row in detail.itertuples(index=False):
        for name in CLASS_NAMES:
            count = int(getattr(row, f"{name}_count"))
            if count == 0:
                warnings.append(
                    f"NO {name.upper()} examples: {row.species}/{row.target_name}/{row.split}"
                )
            elif count < small_class_min:
                warnings.append(
                    f"small {name} class ({count} < {small_class_min}): "
                    f"{row.species}/{row.target_name}/{row.split}"
                )
    return warnings


def print_distribution(detail: pd.DataFrame, small_class_min: int) -> None:
    columns = [
        "species", "tissue_index", "target_name", "split", "usable", "missing_invalid",
        "low_count", "low_percent", "medium_count", "medium_percent", "high_count", "high_percent",
    ]
    printable = detail[columns].copy()
    for column in ("low_percent", "medium_percent", "high_percent"):
        printable[column] = printable[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.2f}")
    print(printable.to_string(index=False))
    warnings = warning_messages(detail, small_class_min)
    if warnings:
        print("\n" + "!" * 88, file=sys.stderr)
        print(f"WARNING: {len(warnings)} absent or very small class cells detected", file=sys.stderr)
        for message in warnings:
            print(f"  - {message}", file=sys.stderr)
        print("!" * 88, file=sys.stderr)


def build(args: argparse.Namespace) -> int:
    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    log2_root = args.log2_input.expanduser().resolve() if args.log2_input else None
    reference_root = args.reference_pgb.expanduser().resolve()
    embed_root = args.embed_root.expanduser().resolve()
    if input_root == output_root:
        raise DatasetError("output must differ from the raw-TPM input directory")
    if not input_root.is_dir():
        raise DatasetError(f"raw-TPM input directory is missing: {input_root}")
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise DatasetError(f"output directory is nonempty: {output_root}; pass --force to overwrite generated files")

    detail_rows: list[dict[str, object]] = []
    species_manifest: dict[str, object] = OrderedDict()
    for species in args.species:
        split_manifest: dict[str, object] = OrderedDict()
        expected_width = None
        targets = None
        for split in SPLITS:
            description = f"{species}/{split}"
            source_path = input_root / species / f"{split}.parquet"
            if not source_path.is_file():
                raise DatasetError(f"source split is missing: {source_path}")
            source = pd.read_parquet(source_path)
            required = {"sequence", "name", "labels"}
            missing = required - set(source.columns)
            if missing:
                raise DatasetError(f"{description}: missing source columns {sorted(missing)}")
            if source["name"].duplicated().any():
                raise DatasetError(f"{description}: duplicate gene/sample IDs")
            verify_reference_alignment(source, reference_root / species / f"{split}.parquet", description)
            tpm = stack_vectors(source["labels"], description)
            if expected_width is None:
                expected_width = tpm.shape[1]
                targets = tissue_targets(species, expected_width)
            elif tpm.shape[1] != expected_width:
                raise DatasetError(f"{description}: tissue target count changed across splits")

            classes, names, valid = assign_classes(tpm, args.low_threshold, args.high_threshold)
            output = source.copy()
            output["tpm_labels"] = [row.astype(np.float32) for row in tpm]
            if log2_root:
                log2_path = log2_root / species / f"{split}.parquet"
                if not log2_path.is_file():
                    raise DatasetError(f"log2-TPM companion split is missing: {log2_path}")
                log_frame = pd.read_parquet(log2_path, columns=["name", "labels"])
                if not np.array_equal(log_frame["name"].astype(str), source["name"].astype(str)):
                    raise DatasetError(f"{description}: log2-TPM gene IDs or row order differ")
                log_values = stack_vectors(log_frame["labels"], f"{description} log2 TPM")
                if log_values.shape != tpm.shape:
                    raise DatasetError(f"{description}: raw/log2 TPM shapes differ")
                finite = np.isfinite(tpm) & np.isfinite(log_values) & (tpm >= 0)
                if finite.any() and not np.allclose(
                    log_values[finite], np.log2(tpm[finite] + 1.0), rtol=2e-5, atol=2e-5
                ):
                    raise DatasetError(f"{description}: companion values are not log2(TPM + 1)")
                output["log2_tpm_labels"] = [row.astype(np.float32) for row in log_values]
            output["class_labels"] = [row for row in classes]
            output["class_names"] = [row.tolist() for row in names]
            output["valid_tpm"] = [row for row in valid]
            output["species"] = species
            output["split"] = split
            output["embedding_row"] = np.arange(len(output), dtype=np.int64)

            destination = output_root / species / f"{split}.parquet"
            if not args.dry_run:
                write_parquet_atomic(output, destination)
            detail_rows.extend(distribution_rows(species, split, classes, valid, targets))
            split_manifest[split] = {
                "rows": len(source),
                "targets": tpm.shape[1],
                "name_sha256": names_digest(source["name"]),
                "source_path": str(source_path),
                "output_path": str(destination),
                "missing_invalid_tpm": int((~valid).sum()),
            }
        species_manifest[species] = {
            "source_dataset": SOURCE_DATASETS[species],
            "targets": targets,
            "splits": split_manifest,
        }

    detail = pd.DataFrame(detail_rows)
    tissue_summary, overall = aggregate_distribution(detail)
    print_distribution(detail, args.small_class_min)
    print("\nOverall class distribution (each gene × tissue value is one target):")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    warnings = warning_messages(detail, args.small_class_min)
    manifest = {
        "format": "pgb-tpm-tissue-classification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_raw_tpm": str(input_root),
        "source_log2_tpm": str(log2_root) if log2_root else None,
        "reference_pgb": str(reference_root),
        "output": str(output_root),
        "split_policy": "copied exactly from source; no rows shuffled, added, or removed",
        "row_alignment": "embedding_row is the zero-based row in the corresponding cached split",
        "thresholds_tpm": {"low_upper_exclusive": args.low_threshold, "high_upper_inclusive": args.high_threshold},
        "class_mapping": {"-1": INVALID_CLASS_NAME, "0": "low", "1": "medium", "2": "high"},
        "class_rules": {
            "low": f"TPM < {args.low_threshold:g}",
            "medium": f"{args.low_threshold:g} <= TPM <= {args.high_threshold:g}",
            "high": f"TPM > {args.high_threshold:g}",
        },
        "invalid_rule": "non-finite or negative TPM; class_labels=-1",
        "embedding_cache_root": str(embed_root),
        "embedding_references": discover_embedding_references(embed_root),
        "species": species_manifest,
        "warnings": warnings,
        "summary_files": {
            "species_tissue_split": "class_distribution_by_tissue_split.csv",
            "species_tissue_all_splits": "class_distribution_by_tissue.csv",
            "overall": "class_distribution_overall.csv",
        },
    }
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        write_text_atomic(detail.to_csv(index=False), output_root / "class_distribution_by_tissue_split.csv")
        write_text_atomic(tissue_summary.to_csv(index=False), output_root / "class_distribution_by_tissue.csv")
        write_text_atomic(overall.to_csv(index=False), output_root / "class_distribution_overall.csv")
        write_text_atomic(json.dumps(manifest, indent=2, allow_nan=False) + "\n", output_root / "manifest.json")
        print(f"\nWrote classification dataset: {output_root}")
    else:
        print("\nDry run: no files written")
    return 0


def inspect(args: argparse.Namespace) -> int:
    dataset = args.dataset.expanduser().resolve()
    path = dataset / "class_distribution_by_tissue_split.csv"
    if not path.is_file():
        raise DatasetError(f"distribution summary is missing: {path}")
    detail = pd.read_csv(path)
    print_distribution(detail, args.small_class_min)
    overall_path = dataset / "class_distribution_overall.csv"
    if overall_path.is_file():
        print("\nOverall class distribution:")
        print(pd.read_csv(overall_path).to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="create the labeled, row-aligned dataset")
    build_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build_parser.add_argument("--log2-input", type=Path, default=DEFAULT_LOG2_INPUT)
    build_parser.add_argument("--reference-pgb", type=Path, default=DEFAULT_REFERENCE)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build_parser.add_argument("--embed-root", type=Path, default=DEFAULT_EMBED_ROOT)
    build_parser.add_argument("--species", nargs="+", choices=SPECIES, default=list(SPECIES))
    build_parser.add_argument("--low-threshold", type=float, default=5.0)
    build_parser.add_argument("--high-threshold", type=float, default=100.0)
    build_parser.add_argument("--small-class-min", type=int, default=20)
    build_parser.add_argument("--dry-run", action="store_true")
    build_parser.add_argument("--force", action="store_true")
    build_parser.set_defaults(func=build)
    inspect_parser = subparsers.add_parser("inspect", help="print an existing dataset's class summaries")
    inspect_parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT)
    inspect_parser.add_argument("--small-class-min", type=int, default=20)
    inspect_parser.set_defaults(func=inspect)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        return args.func(args)
    except (DatasetError, OSError, ValueError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
