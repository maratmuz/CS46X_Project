#!/usr/bin/env python3
"""Compare untouched and fine-tuned heads on identical held-out test sets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PGB_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc")
SUPPLEMENTAL_METRICS = (
    "low_call_rate", "medium_call_rate", "high_call_rate", "mean_probability_low"
)


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def comparison_fields(metrics: tuple[str, ...]) -> list[str]:
    return [
        item for metric in metrics
        for item in (f"baseline_{metric}", f"finetuned_{metric}", f"delta_{metric}")
    ]


def compare_values(
    baseline: dict[str, object], finetuned: dict[str, object], metrics: tuple[str, ...]
) -> dict[str, float]:
    result = {}
    for metric in metrics:
        old = float(baseline[metric])
        new = float(finetuned[metric])
        result[f"baseline_{metric}"] = old
        result[f"finetuned_{metric}"] = new
        result[f"delta_{metric}"] = new - old
    return result


def compare_pgb(source_sweep: Path, fine_sweep: Path, output_dir: Path) -> None:
    rows = []
    for fine_run in sorted((fine_sweep / "runs").iterdir()):
        config_path = fine_run / "run_config.json"
        if not config_path.is_file():
            continue
        config = read_json(config_path)
        source_run = Path(str(config["source_run_dir"]))
        if source_run.parent != source_sweep / "runs":
            raise ValueError(f"source run is outside the expected sweep: {source_run}")
        for species in config["species"]:
            old_path = source_run / "evaluation" / species / "test_metrics.json"
            if not old_path.is_file():
                old_path = source_run / species / "test_metrics.json"
            new_path = fine_run / "evaluation" / species / "test_metrics.json"
            old = read_json(old_path)
            new = read_json(new_path)
            rows.append({
                "head": f"{config['model']}/{config['size']}",
                "species": species,
                **compare_values(old["overall"], new["overall"], PGB_METRICS),
            })
        old_summary_path = source_run / "evaluation" / "summary.json"
        if not old_summary_path.is_file():
            old_summary_path = source_run / "summary.json"
        new_summary_path = fine_run / "evaluation" / "summary.json"
        if old_summary_path.is_file() and new_summary_path.is_file():
            old_summary = read_json(old_summary_path)
            new_summary = read_json(new_summary_path)
            for label, key in (
                ("POOLED", "pooled_gene_tissue_values"),
                ("MACRO_SPECIES_MEAN", "macro_mean_across_species"),
            ):
                rows.append({
                    "head": f"{config['model']}/{config['size']}",
                    "species": label,
                    **compare_values(old_summary[key], new_summary[key], PGB_METRICS),
                })
    if not rows:
        raise ValueError("no fine-tuned PGB evaluation results were found")
    write_csv(
        output_dir / "baseline_vs_finetuned_pgb_test.csv", rows,
        ["head", "species", *comparison_fields(PGB_METRICS)],
    )


def read_table(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["head"], row["locus_set"]): row for row in rows}


def compare_supplemental(fine_sweep: Path, output_dir: Path) -> None:
    old_path = fine_sweep / "baseline_supplemental_test" / "metrics_by_model_category.csv"
    new_path = fine_sweep / "supplemental_test" / "metrics_by_model_category.csv"
    if not old_path.is_file():
        return
    baseline = read_table(old_path)
    finetuned = read_table(new_path)
    if baseline.keys() != finetuned.keys():
        raise ValueError("baseline and fine-tuned supplemental result groups differ")
    rows = []
    for head, locus_set in sorted(baseline):
        rows.append({
            "head": head,
            "locus_set": locus_set,
            **compare_values(baseline[(head, locus_set)], finetuned[(head, locus_set)], SUPPLEMENTAL_METRICS),
        })
    write_csv(
        output_dir / "baseline_vs_finetuned_supplemental_test.csv", rows,
        ["head", "locus_set", *comparison_fields(SUPPLEMENTAL_METRICS)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sweep-dir", type=Path, required=True)
    parser.add_argument("--fine-sweep-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_sweep_dir.expanduser().resolve()
    fine = args.fine_sweep_dir.expanduser().resolve()
    output = fine / "comparison"
    compare_pgb(source, fine, output)
    compare_supplemental(fine, output)
    print(f"Wrote held-out baseline versus fine-tuned comparisons: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
