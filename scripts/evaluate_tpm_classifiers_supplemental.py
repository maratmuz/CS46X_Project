#!/usr/bin/env python3
"""Evaluate trained TPM classifiers on unseen pseudogene/intergenic loci.

The classifier heads were trained only on PGB genes. Supplemental loci have no
tissue TPM measurements, so this script reports *presumed-negative low-call
rates* rather than claiming ordinary multiclass accuracy. Frozen heads consume
previously cached supplemental embeddings; the DeepCRE-style CNN consumes the
same raw 6-kb sequences directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"mpl-{os.getuid()}"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / f"xdg-{os.getuid()}"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import train_deepcre_tpm_classifier as deepcre
import train_tpm_classifiers as common


DEFAULT_SWEEP = Path("/nfs/hpc/share/evo2_shared/tpm-classification-sweeps/tpm_balanced_v1")
DEFAULT_DATASET = Path("/nfs/hpc/share/evo2_shared/datasets/supplemental_loci_parquet")
DEFAULT_EMBED_ROOT = Path("/nfs/hpc/share/evo2_shared/frozen-embeddings/supplemental")
LOCUS_SETS = ("pseudogenes", "intergenic")
SPLITS = ("train", "validation", "test")
CLASS_NAMES = common.CLASS_NAMES


class SupplementalError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SupplementalError(f"required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(runs_root: Path, selected: Sequence[str] | None) -> list[tuple[Path, dict[str, object]]]:
    wanted = set(selected or [])
    found = []
    for run_dir in sorted(runs_root.iterdir()):
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            continue
        config = read_json(config_path)
        head = f"{config['model']}/{config['size']}"
        if wanted and head not in wanted:
            continue
        if not (run_dir / "summary.json").is_file():
            raise SupplementalError(f"classifier run is incomplete: {run_dir}")
        found.append((run_dir, config))
    if wanted:
        missing = wanted.difference(f"{config['model']}/{config['size']}" for _, config in found)
        if missing:
            raise SupplementalError(f"requested classifier runs were not found: {sorted(missing)}")
    if not found:
        raise SupplementalError(f"no completed classifier runs found under {runs_root}")
    return found


def supplemental_parquet(dataset_root: Path, locus_set: str, species: str, split: str) -> Path:
    path = dataset_root / locus_set / species / f"{split}.parquet"
    if not path.is_file():
        raise SupplementalError(f"supplemental dataset split is missing: {path}")
    return path


def embedding_path(
    embed_root: Path,
    locus_set: str,
    model: str,
    size: str,
    species: str,
    split: str,
    layer: int | None,
) -> Path:
    filename = f"{split}.pt" if layer is None else f"{split}-L{layer}.pt"
    return embed_root / locus_set / model / size / species / filename


def missing_embedding_paths(
    runs: Sequence[tuple[Path, dict[str, object]]],
    embed_root: Path,
    locus_sets: Sequence[str],
    splits: Sequence[str],
) -> list[Path]:
    missing = []
    for run_dir, config in runs:
        if config["model"] == "deepcre":
            continue
        for species in config["species"]:
            checkpoint = torch.load(run_dir / species / "best.pt", map_location="cpu", weights_only=False)
            layer = checkpoint.get("layer")
            for locus_set in locus_sets:
                for split in splits:
                    path = embedding_path(
                        embed_root, locus_set, config["model"], config["size"], species, split, layer
                    )
                    if not path.is_file():
                        missing.append(path)
    return missing


def infer_frozen_head(
    checkpoint_path: Path, embedding_file: Path, expected_rows: int, device: str, batch_size: int
) -> torch.Tensor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cache = torch.load(embedding_file, map_location="cpu", weights_only=True)
    if "X" not in cache:
        raise SupplementalError(f"embedding cache has no X tensor: {embedding_file}")
    features = cache["X"].float()
    if len(features) != expected_rows:
        raise SupplementalError(
            f"row mismatch: parquet={expected_rows}, embeddings={len(features)} ({embedding_file})"
        )
    model = common.TissueClassifierHead(
        checkpoint["in_dim"], checkpoint["n_tissues"], checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return common.infer_logits(
        model, features, checkpoint["feature_mean"], checkpoint["feature_std"], device, batch_size
    )


def infer_deepcre_head(
    checkpoint_path: Path, sequences: Sequence[str], device: str, batch_size: int
) -> torch.Tensor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if any(len(sequence) != checkpoint["sequence_length"] for sequence in sequences):
        raise SupplementalError(
            f"sequence length differs from DeepCRE checkpoint contract: {checkpoint['sequence_length']}"
        )
    model = deepcre.DeepCREClassifier(
        checkpoint["sequence_length"], checkpoint["n_tissues"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    data = {"sequences": list(sequences)}
    return deepcre.infer_logits(model, data, device, batch_size)


def prediction_frame(
    head: str,
    locus_set: str,
    species: str,
    split: str,
    names: Sequence[str],
    targets: Sequence[dict[str, object]],
    logits: torch.Tensor,
) -> pd.DataFrame:
    probabilities = torch.softmax(logits, dim=-1).numpy()
    predicted = probabilities.argmax(axis=2)
    n_loci, n_tissues = predicted.shape
    if n_loci != len(names) or n_tissues != len(targets):
        raise SupplementalError(f"prediction shape does not match locus/tissue metadata for {head}/{species}")
    return pd.DataFrame(
        {
            "head": head,
            "locus_set": locus_set,
            "species": species,
            "split": split,
            "locus_id": np.repeat(np.asarray(names, dtype=object), n_tissues),
            "source_row": np.repeat(np.arange(n_loci, dtype=np.int64), n_tissues),
            "tissue_index": np.tile(np.arange(n_tissues, dtype=np.int16), n_loci),
            "target_name": np.tile([target["target_name"] for target in targets], n_loci),
            "tissue": np.tile([target["tissue"] for target in targets], n_loci),
            "presumed_class": "low",
            "predicted_class": predicted.reshape(-1).astype(np.int8),
            "predicted_class_name": np.asarray(CLASS_NAMES, dtype=object)[predicted].reshape(-1),
            "probability_low": probabilities[:, :, 0].reshape(-1),
            "probability_medium": probabilities[:, :, 1].reshape(-1),
            "probability_high": probabilities[:, :, 2].reshape(-1),
        }
    )


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    predicted = frame["predicted_class"].to_numpy(dtype=np.int64)
    probabilities = frame[["probability_low", "probability_medium", "probability_high"]].to_numpy()
    low_probability = np.clip(probabilities[:, 0], 1e-12, 1.0)
    locus_keys = ["species", "split", "locus_id"]
    per_locus_low = frame.assign(is_low=predicted == 0).groupby(locus_keys, sort=False)["is_low"]
    counts = np.bincount(predicted, minlength=3)
    result: dict[str, object] = {
        "n_loci": int(frame[locus_keys].drop_duplicates().shape[0]),
        "n_tissue_decisions": int(len(frame)),
        "low_count": int(counts[0]),
        "medium_count": int(counts[1]),
        "high_count": int(counts[2]),
        "low_call_rate": float((predicted == 0).mean()),
        "medium_call_rate": float((predicted == 1).mean()),
        "high_call_rate": float((predicted == 2).mean()),
        "mean_probability_low": float(probabilities[:, 0].mean()),
        "mean_probability_medium": float(probabilities[:, 1].mean()),
        "mean_probability_high": float(probabilities[:, 2].mean()),
        "low_negative_log_likelihood": float(-np.log(low_probability).mean()),
        "low_brier_score": float(np.square(probabilities - np.array([1.0, 0.0, 0.0])).sum(axis=1).mean()),
        "all_tissues_low_locus_rate": float(per_locus_low.all().mean()),
        "majority_tissues_low_locus_rate": float((per_locus_low.mean() > 0.5).mean()),
    }
    return result


def grouped_summary(
    predictions: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(list(group_columns), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(group_columns, keys)), **summarize(group)})
    return pd.DataFrame(rows)


def write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def save_figure(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_overall_bars(wide: pd.DataFrame, destination: Path) -> None:
    heads = wide["head"].tolist()
    x = np.arange(len(heads))
    figure, axis = plt.subplots(figsize=(14, 7))
    width = 0.36
    axis.bar(x - width / 2, 100 * wide["pseudogenes"], width, label="Pseudogene")
    axis.bar(x + width / 2, 100 * wide["intergenic"], width, label="Intergenic")
    axis.set_xticks(x, heads, rotation=40, ha="right")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Predicted low (%)")
    axis.set_title("Unseen supplemental loci called low expression", fontsize=17, fontweight="bold", loc="left")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    save_figure(figure, destination)


def render_primary_table(wide: pd.DataFrame, destination: Path) -> None:
    def percentage(value: float) -> str:
        return "—" if not np.isfinite(value) else f"{100 * value:.1f}%"

    display = pd.DataFrame(
        {
            "Model": wide["head"],
            "Pseudogene": wide["pseudogenes"].map(percentage),
            "Intergenic": wide["intergenic"].map(percentage),
        }
    )
    height = max(4, 0.58 * len(display) + 1.7)
    figure, axis = plt.subplots(figsize=(12, height))
    axis.axis("off")
    table = axis.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.7)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f5f88")
        elif row % 2 == 0:
            cell.set_facecolor("#edf3f8")
        if column == 0:
            cell.set_text_props(ha="left", weight="bold" if row else "bold")
    axis.set_title(
        "Low-expression calls on unseen loci", fontsize=18, fontweight="bold", loc="left", pad=18
    )
    save_figure(figure, destination)


def plot_species(overall_species: pd.DataFrame, destination: Path) -> None:
    species = list(dict.fromkeys(overall_species["species"]))
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for axis, locus_set in zip(axes, LOCUS_SETS):
        subset = overall_species[overall_species["locus_set"] == locus_set]
        for head, group in subset.groupby("head", sort=False):
            values = group.set_index("species").reindex(species)["low_call_rate"]
            axis.plot(species, 100 * values, marker="o", linewidth=1.6, label=head)
        axis.set_title(locus_set.capitalize(), fontweight="bold")
        axis.set_ylim(0, 100)
        axis.set_ylabel("Predicted low (%)")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.2)
    axes[1].legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("Supplemental low-call rate by species", fontsize=18, fontweight="bold", x=0.02, ha="left")
    figure.tight_layout(rect=(0, 0, 0.88, 0.95))
    save_figure(figure, destination)


def write_report(output_dir: Path, overall: pd.DataFrame, species: pd.DataFrame) -> None:
    pivot = overall.pivot(index="head", columns="locus_set", values="low_call_rate").reset_index()
    for locus_set in LOCUS_SETS:
        if locus_set not in pivot:
            pivot[locus_set] = np.nan
    pivot = pivot[["head", *LOCUS_SETS]]
    pivot.to_csv(output_dir / "low_call_rate_table.csv", index=False)
    plot_overall_bars(pivot, output_dir / "low_call_rates")
    render_primary_table(pivot, output_dir / "low_call_rate_table")
    plot_species(species, output_dir / "low_call_rates_by_species")

    markdown = [
        "# Supplemental-locus classification report",
        "",
        "The table reports the percentage of tissue-specific decisions assigned to the low-expression class.",
        "",
        "| Model | Pseudogene | Intergenic |",
        "| --- | ---: | ---: |",
    ]
    def markdown_percentage(value: float) -> str:
        return "—" if not np.isfinite(value) else f"{100 * value:.1f}%"

    for row in pivot.itertuples(index=False):
        markdown.append(
            f"| {row.head} | {markdown_percentage(row.pseudogenes)} | "
            f"{markdown_percentage(row.intergenic)} |"
        )
    markdown.extend(
        [
            "",
            "## Interpretation",
            "",
            "These loci were never used for classifier-head training, class weighting, early stopping, or model selection. "
            "All supplemental train/validation/test partitions are pooled because those labels refer to a separate "
            "locus-type dataset and none participated in expression-head fitting.",
            "",
            "This claim applies to the task-specific heads. The pretraining corpora of the underlying foundation "
            "models cannot generally be audited well enough to claim that the backbones never encountered related "
            "genomic sequence. DeepCRE is trained end-to-end here, and its task-specific CNN saw only the PGB splits.",
            "",
            "The supplemental data have no measured tissue TPM. Therefore, `low_call_rate` is a presumed-negative "
            "behavioral test, not verified biological accuracy. This caveat is especially important for pseudogenes, "
            "some of which can be transcribed. Intergenic loci are likewise sampled windows rather than assayed "
            "zero-expression targets. Medium/high calls quantify expression-like behavior on these sequences.",
            "",
            "`metrics_by_model_category.csv` contains pooled decision and locus-level metrics. Species, tissue, and "
            "split breakdowns are in the corresponding CSV files. Raw probabilities are retained as Parquet files "
            "under `predictions/` for auditing and alternative summaries.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> int:
    sweep_dir = args.sweep_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    embed_root = args.embed_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    runs = discover_runs(sweep_dir / "runs", args.models)
    missing = missing_embedding_paths(runs, embed_root, args.locus_sets, args.splits)
    if missing:
        sample = "\n".join(f"  - {path}" for path in missing[:12])
        raise SupplementalError(
            f"{len(missing)} supplemental embedding caches are missing. Run the HPC wrapper first.\n{sample}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SupplementalError("CUDA requested but unavailable; pass --device cpu")
    pgb_manifest = common.load_manifest(Path(runs[0][1]["dataset_root"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frames = []

    for run_dir, config in runs:
        model_name, size = str(config["model"]), str(config["size"])
        head = f"{model_name}/{size}"
        for species_name in config["species"]:
            checkpoint_path = run_dir / species_name / "best.pt"
            targets = pgb_manifest["species"][species_name]["targets"]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            for locus_set in args.locus_sets:
                for split in args.splits:
                    destination = output_dir / "predictions" / model_name / size / locus_set / species_name / f"{split}.parquet"
                    if args.resume and destination.is_file():
                        prediction_frames.append(pd.read_parquet(destination))
                        print(f"Skipping completed predictions: {head}/{locus_set}/{species_name}/{split}")
                        continue
                    columns = ["name", "sequence"] if model_name == "deepcre" else ["name"]
                    source = pd.read_parquet(
                        supplemental_parquet(dataset_root, locus_set, species_name, split), columns=columns
                    )
                    if model_name == "deepcre":
                        logits = infer_deepcre_head(
                            checkpoint_path, source["sequence"].astype(str).str.upper().tolist(),
                            args.device, args.deepcre_batch_size,
                        )
                    else:
                        cache_path = embedding_path(
                            embed_root, locus_set, model_name, size, species_name, split,
                            checkpoint.get("layer"),
                        )
                        logits = infer_frozen_head(
                            checkpoint_path, cache_path, len(source), args.device, args.batch_size
                        )
                    predictions = prediction_frame(
                        head, locus_set, species_name, split, source["name"].astype(str).tolist(), targets, logits
                    )
                    write_parquet(predictions, destination)
                    prediction_frames.append(predictions)
                    print(
                        f"{head:20s} {locus_set:12s} {species_name:24s} {split:10s} "
                        f"low={100 * (predictions['predicted_class'] == 0).mean():6.2f}%",
                        flush=True,
                    )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    overall = grouped_summary(all_predictions, ["head", "locus_set"])
    by_species = grouped_summary(all_predictions, ["head", "locus_set", "species"])
    macro_species = (
        by_species.groupby(["head", "locus_set"], as_index=False)["low_call_rate"]
        .mean()
        .rename(columns={"low_call_rate": "macro_species_low_call_rate"})
    )
    overall = overall.merge(macro_species, on=["head", "locus_set"], how="left", validate="one_to_one")
    by_tissue = grouped_summary(
        all_predictions, ["head", "locus_set", "species", "tissue_index", "target_name", "tissue"]
    )
    by_split = grouped_summary(all_predictions, ["head", "locus_set", "species", "split"])
    overall.to_csv(output_dir / "metrics_by_model_category.csv", index=False)
    by_species.to_csv(output_dir / "metrics_by_model_species_category.csv", index=False)
    by_tissue.to_csv(output_dir / "metrics_by_model_species_tissue_category.csv", index=False)
    by_split.to_csv(output_dir / "metrics_by_model_species_split_category.csv", index=False)
    write_report(output_dir, overall, by_species)
    common.write_json_atomic(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sweep_dir": str(sweep_dir),
            "dataset_root": str(dataset_root),
            "embed_root": str(embed_root),
            "heads": [f"{config['model']}/{config['size']}" for _, config in runs],
            "locus_sets": args.locus_sets,
            "splits_pooled": args.splits,
            "device": args.device,
            "interpretation": "presumed-negative low-call behavior; supplemental tissue TPM is unavailable",
            "leakage_control": "supplemental data unused for training, weighting, early stopping, and selection",
        },
        output_dir / "evaluation_config.json",
    )
    print(f"\nWrote supplemental classifier evaluation: {output_dir}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--embed-root", type=Path, default=DEFAULT_EMBED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SWEEP / "supplemental_evaluation")
    parser.add_argument("--models", nargs="+", default=None, help="Optional subset such as evo2/7b deepcre/cnn")
    parser.add_argument("--locus-sets", nargs="+", choices=LOCUS_SETS, default=list(LOCUS_SETS))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--deepcre-batch-size", type=int, default=128)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return evaluate(parse_args(argv))
    except (SupplementalError, common.ClassifierError, OSError, ValueError, KeyError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
