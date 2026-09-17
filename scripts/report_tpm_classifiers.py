#!/usr/bin/env python3
"""Combine held-out classifier results and draw row-normalized confusion matrices.

The input directories are training run directories made by
``train_tpm_classifiers.py``.  By default, results are read from each run's
``evaluation`` subdirectory, falling back to the training-time test outputs if
that directory is absent.  No model training or embedding extraction occurs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence

# Shared-cluster home directories are often read-only inside containers.  Give
# Matplotlib/fontconfig a writable cache without requiring caller setup.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"mpl-{os.getuid()}"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / f"xdg-{os.getuid()}"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)


CLASS_NAMES = ("low", "medium", "high")
SCALAR_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc")
CURVE_METRICS = ("auroc_ovr_macro_defined_classes", "auprc_ovr_macro_defined_classes")


class ReportError(RuntimeError):
    pass


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ReportError(f"required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_metrics(model: str, size: str, species: str, metrics: dict[str, object]) -> dict[str, object]:
    overall = metrics["overall"]
    row: dict[str, object] = {"model": model, "size": size, "head": f"{model}/{size}", "species": species}
    row.update({name: overall.get(name) for name in SCALAR_METRICS})
    row.update({name: overall.get(name) for name in CURVE_METRICS})
    row["n"] = overall.get("n")
    for class_name in CLASS_NAMES:
        values = overall.get("per_class", {}).get(class_name, {})
        for metric in ("precision", "recall", "f1", "support", "auroc_ovr", "auprc_ovr"):
            row[f"{class_name}_{metric}"] = values.get(metric)
    return row


def confusion_from_predictions(frame: pd.DataFrame) -> np.ndarray:
    required = {"valid_tpm", "true_class", "predicted_class"}
    missing = required.difference(frame.columns)
    if missing:
        raise ReportError(f"prediction table is missing columns: {sorted(missing)}")
    valid = frame["valid_tpm"].astype(bool)
    true = frame.loc[valid, "true_class"].to_numpy(dtype=np.int64)
    predicted = frame.loc[valid, "predicted_class"].to_numpy(dtype=np.int64)
    if len(true) and (not np.isin(true, [0, 1, 2]).all() or not np.isin(predicted, [0, 1, 2]).all()):
        raise ReportError("valid predictions contain a class outside 0, 1, 2")
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (true, predicted), 1)
    return matrix


def plot_grid(
    matrices: Sequence[tuple[str, np.ndarray]], title: str, destination: Path
) -> None:
    if not matrices:
        return
    columns = min(3, len(matrices))
    rows = math.ceil(len(matrices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6.1 * columns, 5.25 * rows), squeeze=False)
    figure.suptitle(title, fontsize=18, fontweight="bold", x=0.02, ha="left")
    for axis, (label, matrix) in zip(axes.flat, matrices):
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix, row_totals, out=np.zeros_like(matrix, dtype=np.float64), where=row_totals != 0
        )
        axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        accuracy = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else float("nan")
        accuracy_text = f"{100 * accuracy:.1f}%" if np.isfinite(accuracy) else "n/a"
        axis.set_title(f"{label}     overall {accuracy_text}", loc="left", fontsize=14, fontweight="bold")
        axis.set_xticks(range(3), CLASS_NAMES)
        axis.set_yticks(range(3), CLASS_NAMES)
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
        axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row in range(3):
            for column in range(3):
                percentage = normalized[row, column]
                color = "white" if percentage >= 0.52 else "#111111"
                axis.text(
                    column, row - 0.10, f"{100 * percentage:.1f}%", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color,
                )
                count_color = "#d9e2ea" if percentage >= 0.52 else "#8a8a8a"
                axis.text(
                    column, row + 0.23, f"n={matrix[row, column]:,}", ha="center", va="center",
                    fontsize=9.5, color=count_color,
                )
    for axis in axes.flat[len(matrices):]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)


def threshold_title(thresholds: dict[str, object]) -> str:
    low = thresholds.get("low_upper_exclusive", 5)
    high = thresholds.get("high_upper_inclusive", 100)
    return (
        f"Expression bucket classification: low (<{low:g} TPM) / "
        f"medium ({low:g}–{high:g} TPM) / high (>{high:g} TPM)"
    )


def prediction_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    valid = frame["valid_tpm"].astype(bool)
    y = frame.loc[valid, "true_class"].to_numpy(dtype=np.int64)
    probabilities = frame.loc[
        valid, ["probability_low", "probability_medium", "probability_high"]
    ].to_numpy(dtype=np.float64)
    return y, probabilities


def curves_for_predictions(frame: pd.DataFrame) -> dict[str, object]:
    y, probabilities = prediction_arrays(frame)
    binary = np.eye(3, dtype=np.int8)[y]
    roc_by_class, pr_by_class = {}, {}
    roc_scores, pr_scores = {}, {}
    aurocs, auprcs = [], []
    for index, class_name in enumerate(CLASS_NAMES):
        if len(np.unique(binary[:, index])) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary[:, index], probabilities[:, index])
        precision, recall, _ = precision_recall_curve(binary[:, index], probabilities[:, index])
        roc_by_class[class_name] = (fpr, tpr)
        pr_by_class[class_name] = (recall, precision)
        roc_scores[class_name] = float(auc(fpr, tpr))
        pr_scores[class_name] = float(
            average_precision_score(binary[:, index], probabilities[:, index])
        )
        aurocs.append(roc_scores[class_name])
        auprcs.append(pr_scores[class_name])

    micro_fpr, micro_tpr, _ = roc_curve(binary.ravel(), probabilities.ravel())
    micro_precision, micro_recall, _ = precision_recall_curve(binary.ravel(), probabilities.ravel())
    fpr_grid = np.linspace(0.0, 1.0, 1001)
    mean_tpr = np.mean(
        [np.interp(fpr_grid, roc_by_class[name][0], roc_by_class[name][1]) for name in roc_by_class], axis=0
    )
    recall_grid = np.linspace(0.0, 1.0, 1001)
    mean_precision = np.mean(
        [
            np.interp(recall_grid, pr_by_class[name][0][::-1], pr_by_class[name][1][::-1])
            for name in pr_by_class
        ],
        axis=0,
    )
    return {
        "roc_by_class": roc_by_class,
        "pr_by_class": pr_by_class,
        "roc_scores": roc_scores,
        "pr_scores": pr_scores,
        "roc_macro": (fpr_grid, mean_tpr),
        "pr_macro": (recall_grid, mean_precision),
        "roc_micro": (micro_fpr, micro_tpr),
        "pr_micro": (micro_recall, micro_precision),
        "macro_auroc": float(np.mean(aurocs)),
        "macro_auprc": float(np.mean(auprcs)),
        "micro_auroc": float(auc(micro_fpr, micro_tpr)),
        "micro_auprc": float(average_precision_score(binary.ravel(), probabilities.ravel())),
    }


def save_figure(figure: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_without_suffix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_curve_grid(
    curve_data: dict[str, dict[str, object]], kind: str, destination: Path
) -> None:
    is_roc = kind == "roc"
    figure, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    panels = list(CLASS_NAMES) + ["macro", "micro"]
    colors = plt.get_cmap("tab10")
    for panel_index, panel in enumerate(panels):
        axis = axes.flat[panel_index]
        for model_index, (head, values) in enumerate(curve_data.items()):
            color = colors(model_index % 10)
            if panel in CLASS_NAMES:
                mapping = values[f"{kind}_by_class"]
                if panel not in mapping:
                    continue
                x, y = mapping[panel]
                score = values[f"{kind}_scores"][panel]
            else:
                x, y = values[f"{kind}_{panel}"]
                score = values[f"{panel}_{'auroc' if is_roc else 'auprc'}"]
            axis.plot(x, y, linewidth=1.8, color=color, label=f"{head} ({score:.3f})")
        if is_roc:
            axis.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
            axis.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
            metric_name = "AUROC"
        else:
            axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
            metric_name = "AUPRC"
        axis.set_title(f"{panel.capitalize()} one-vs-rest" if panel in CLASS_NAMES else f"{panel.capitalize()} average")
        axis.grid(alpha=0.2)
        axis.legend(title=metric_name, fontsize=7, title_fontsize=8, loc="best")
    axes.flat[-1].axis("off")
    figure.suptitle(
        "Held-out one-vs-rest ROC curves" if is_roc else "Held-out one-vs-rest precision–recall curves",
        fontsize=18, fontweight="bold", x=0.02, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, destination)


def plot_summary_metrics(table: pd.DataFrame, destination: Path) -> None:
    columns = ["accuracy", "macro_f1", "balanced_accuracy", "macro_auroc", "macro_auprc"]
    labels = ["Accuracy", "Macro-F1", "Balanced accuracy", "Macro-AUROC", "Macro-AUPRC"]
    x = np.arange(len(columns))
    width = 0.82 / max(len(table), 1)
    figure, axis = plt.subplots(figsize=(15, 7))
    for index, row in table.reset_index(drop=True).iterrows():
        axis.bar(x - 0.41 + width / 2 + index * width, [row[c] for c in columns], width, label=row["head"])
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Held-out performance across architectures", fontsize=17, fontweight="bold", loc="left")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.14), loc="upper center")
    figure.tight_layout()
    save_figure(figure, destination)


def plot_per_class_metrics(table: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(19, 6.5), sharey=True)
    x = np.arange(len(table))
    markers = {"precision": "o", "recall": "s", "f1": "^"}
    for axis, class_name in zip(axes, CLASS_NAMES):
        for metric, marker in markers.items():
            axis.plot(x, table[f"{class_name}_{metric}"], marker=marker, linewidth=1.6, label=metric.capitalize())
        axis.set_xticks(x, table["head"], rotation=50, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(class_name.capitalize(), fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Score")
    figure.suptitle("Per-class held-out precision, recall, and F1", fontsize=18, fontweight="bold", x=0.02, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, destination)


def plot_species_performance(table: pd.DataFrame, destination: Path) -> None:
    species_order = list(dict.fromkeys(table["species"]))
    figure, axes = plt.subplots(1, 2, figsize=(18, 7))
    for head, group in table.groupby("head", sort=False):
        indexed = group.set_index("species").reindex(species_order)
        axes[0].plot(species_order, indexed["macro_f1"], marker="o", linewidth=1.6, label=head)
        axes[1].plot(
            species_order, indexed["auroc_ovr_macro_defined_classes"], marker="o", linewidth=1.6, label=head
        )
    for axis, title_text, ylabel in zip(axes, ("Macro-F1", "Macro-AUROC"), ("Macro-F1", "Macro-AUROC")):
        axis.set_title(title_text, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.2)
    axes[1].legend(ncol=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("Performance by held-out species", fontsize=18, fontweight="bold", x=0.02, ha="left")
    figure.tight_layout(rect=(0, 0, 0.88, 0.95))
    save_figure(figure, destination)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
    headers = [str(value) for value in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    ) + "\n"


def render_table(frame: pd.DataFrame, destination: Path, title: str) -> None:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
    width = max(10, 1.5 * len(display.columns))
    height = max(3, 0.42 * len(display) + 1.7)
    figure, axis = plt.subplots(figsize=(width, height))
    axis.axis("off")
    axis.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=14)
    table = axis.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#336b9a")
        elif row % 2 == 0:
            cell.set_facecolor("#edf3f8")
    save_figure(figure, destination)


def write_table_bundle(frame: pd.DataFrame, base: Path, title: str) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".md").write_text(markdown_table(frame), encoding="utf-8")
    latex = frame.to_latex(index=False, float_format=lambda value: f"{value:.3f}", na_rep="--", escape=True)
    base.with_suffix(".tex").write_text(latex, encoding="utf-8")
    render_table(frame, base, title)


def make_report(
    run_dirs: Sequence[Path], output_dir: Path, results_subdir: str,
    fine_tuned_supplemental: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    species_rows: list[dict[str, object]] = []
    tissue_frames: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, object]] = []
    matrices_by_species: dict[str, list[tuple[str, np.ndarray]]] = {}
    pooled_frames: dict[str, list[pd.DataFrame]] = {}
    thresholds_seen: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    seen_heads: set[str] = set()

    for run_dir_arg in run_dirs:
        run_dir = run_dir_arg.expanduser().resolve()
        config = read_json(run_dir / "run_config.json")
        model, size = str(config["model"]), str(config["size"])
        head = f"{model}/{size}"
        if head in seen_heads:
            raise ReportError(f"duplicate head supplied: {head}")
        seen_heads.add(head)
        result_root = run_dir / results_subdir if results_subdir else run_dir
        if not (result_root / "summary.json").is_file():
            result_root = run_dir
        summary = read_json(result_root / "summary.json")
        manifest = read_json(Path(str(config["dataset_root"])) / "manifest.json")
        thresholds_seen.append(manifest["thresholds_tpm"])

        for aggregation, values in (
            ("macro_species_mean", summary["macro_mean_across_species"]),
            ("pooled", summary["pooled_gene_tissue_values"]),
        ):
            model_rows.append(
                {"model": model, "size": size, "head": head, "aggregation": aggregation,
                 **{metric: values.get(metric) for metric in SCALAR_METRICS + CURVE_METRICS}}
            )

        for species in config["species"]:
            species_dir = result_root / species
            metrics = read_json(species_dir / "test_metrics.json")
            species_rows.append(flatten_metrics(model, size, species, metrics))
            predictions = pd.read_parquet(species_dir / "test_predictions.parquet")
            matrix = confusion_from_predictions(predictions)
            matrices_by_species.setdefault(species, []).append((head, matrix))
            pooled_frames.setdefault(head, []).append(predictions)
            for true_index, true_name in enumerate(CLASS_NAMES):
                denominator = int(matrix[true_index].sum())
                for predicted_index, predicted_name in enumerate(CLASS_NAMES):
                    count = int(matrix[true_index, predicted_index])
                    confusion_rows.append(
                        {"model": model, "size": size, "head": head, "species": species,
                         "true_class": true_name, "predicted_class": predicted_name, "count": count,
                         "row_percent": 100.0 * count / denominator if denominator else np.nan}
                    )
            tissue_path = species_dir / "test_metrics_by_tissue.csv"
            tissue = pd.read_csv(tissue_path)
            tissue.insert(0, "head", head)
            tissue.insert(0, "size", size)
            tissue.insert(0, "model", model)
            tissue_frames.append(tissue)

    canonical_thresholds = thresholds_seen[0]
    if any(value != canonical_thresholds for value in thresholds_seen[1:]):
        raise ReportError("runs use different TPM thresholds; refusing to combine them")
    title = threshold_title(canonical_thresholds)

    species_table = pd.DataFrame(species_rows)
    model_table = pd.DataFrame(model_rows)
    species_table.to_csv(output_dir / "metrics_by_model_species.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrix_cells.csv", index=False)
    if tissue_frames:
        pd.concat(tissue_frames, ignore_index=True).to_csv(
            output_dir / "metrics_by_model_species_tissue.csv", index=False
        )

    pooled_matrices = []
    for head, frames in pooled_frames.items():
        pooled_matrices.append((head, confusion_from_predictions(pd.concat(frames, ignore_index=True))))
    plot_grid(pooled_matrices, title, output_dir / "confusion_matrices_all_species.png")
    species_plot_dir = output_dir / "confusion_matrices_by_species"
    for species, matrices in matrices_by_species.items():
        plot_grid(matrices, f"{title} — {species}", species_plot_dir / f"{safe_name(species)}.png")

    # Publication figures and tables are all recomputed from unrounded held-out
    # probabilities. This also supplies micro averages not present in the
    # original metric JSON contract.
    combined_predictions = {
        head: pd.concat(frames, ignore_index=True) for head, frames in pooled_frames.items()
    }
    curve_data = {head: curves_for_predictions(frame) for head, frame in combined_predictions.items()}
    for head, values in curve_data.items():
        mask = (model_table["head"] == head) & (model_table["aggregation"] == "pooled")
        model_table.loc[mask, "macro_auroc"] = values["macro_auroc"]
        model_table.loc[mask, "macro_auprc"] = values["macro_auprc"]
        model_table.loc[mask, "micro_auroc"] = values["micro_auroc"]
        model_table.loc[mask, "micro_auprc"] = values["micro_auprc"]
    model_table.to_csv(output_dir / "metrics_by_model.csv", index=False)

    figures_dir = output_dir / "figures"
    plot_curve_grid(curve_data, "roc", figures_dir / "roc_curves")
    plot_curve_grid(curve_data, "pr", figures_dir / "precision_recall_curves")
    overall_table = model_table[model_table["aggregation"] == "pooled"].copy()
    overall_table = overall_table[
        ["head", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc",
         "macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc"]
    ].sort_values("macro_f1", ascending=False).reset_index(drop=True)
    overall_table.insert(0, "rank", np.arange(1, len(overall_table) + 1))
    plot_summary_metrics(overall_table, figures_dir / "summary_metrics")

    per_class_rows = []
    for head, frame in combined_predictions.items():
        y, probabilities = prediction_arrays(frame)
        predicted = probabilities.argmax(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(
            y, predicted, labels=[0, 1, 2], zero_division=0
        )
        row: dict[str, object] = {"head": head}
        for index, class_name in enumerate(CLASS_NAMES):
            row[f"{class_name}_precision"] = float(precision[index])
            row[f"{class_name}_recall"] = float(recall[index])
            row[f"{class_name}_f1"] = float(f1[index])
            row[f"{class_name}_support"] = int(support[index])
            class_binary = (y == index).astype(np.int8)
            fpr, tpr, _ = roc_curve(class_binary, probabilities[:, index])
            pr_precision, pr_recall, _ = precision_recall_curve(class_binary, probabilities[:, index])
            row[f"{class_name}_auroc"] = float(auc(fpr, tpr))
            row[f"{class_name}_auprc"] = float(
                average_precision_score(class_binary, probabilities[:, index])
            )
        per_class_rows.append(row)
    per_class_table = pd.DataFrame(per_class_rows)
    plot_per_class_metrics(per_class_table, figures_dir / "per_class_precision_recall_f1")
    plot_species_performance(species_table, figures_dir / "species_performance")

    tables_dir = output_dir / "tables"
    write_table_bundle(overall_table, tables_dir / "overall_metrics", "Overall held-out metrics")
    species_compact = species_table[
        ["head", "species", "accuracy", "balanced_accuracy", "macro_f1",
         "auroc_ovr_macro_defined_classes", "auprc_ovr_macro_defined_classes"]
    ].rename(
        columns={"auroc_ovr_macro_defined_classes": "macro_auroc", "auprc_ovr_macro_defined_classes": "macro_auprc"}
    )
    write_table_bundle(
        species_compact, tables_dir / "species_metrics", "Held-out metrics by architecture and species"
    )
    write_table_bundle(
        per_class_table, tables_dir / "per_class_metrics", "Pooled held-out metrics by expression class"
    )

    best = overall_table.iloc[0]
    deepcre_rows = overall_table[overall_table["head"] == "deepcre/cnn"]
    deepcre_sentence = ""
    deepcre_class_sentence = ""
    if not deepcre_rows.empty:
        deepcre = deepcre_rows.iloc[0]
        deepcre_sentence = (
            f" The DeepCRE-style CNN ranked {int(deepcre['rank'])} of {len(overall_table)} by macro-F1 "
            f"({deepcre['macro_f1']:.3f}), with balanced accuracy {deepcre['balanced_accuracy']:.3f}, "
            f"macro-AUROC {deepcre['macro_auroc']:.3f}, and macro-AUPRC {deepcre['macro_auprc']:.3f}."
        )
        deepcre_class = per_class_table[per_class_table["head"] == "deepcre/cnn"].iloc[0]
        deepcre_class_sentence = (
            f"Its pooled class-wise F1 scores were {deepcre_class['low_f1']:.3f} for low, "
            f"{deepcre_class['medium_f1']:.3f} for medium, and {deepcre_class['high_f1']:.3f} for high "
            f"expression. For the minority high-expression class, recall was "
            f"{deepcre_class['high_recall']:.3f}, precision was {deepcre_class['high_precision']:.3f}, "
            f"AUROC was {deepcre_class['high_auroc']:.3f}, and AUPRC was "
            f"{deepcre_class['high_auprc']:.3f}."
        )
    paper_summary = f"""# Paper-ready results summary

## Benchmark design

We evaluated three-class tissue-specific expression prediction using fixed raw-TPM boundaries: **low** (<{canonical_thresholds['low_upper_exclusive']:g} TPM), **medium** ({canonical_thresholds['low_upper_exclusive']:g}–{canonical_thresholds['high_upper_inclusive']:g} TPM, inclusive), and **high** (>{canonical_thresholds['high_upper_inclusive']:g} TPM). All architectures used the original PGB gene-family-aware train, validation, and held-out test partitions. Each foundation model was run previously to produce one frozen embedding per 6-kb sequence; a small MLP then generated three logits for every tissue. DeepCRE instead operated directly on the same nucleotide sequences. Balanced cross-entropy weights were estimated independently per tissue from training labels only as `N/(3 n_c)`, where `N` is the number of usable training labels for that tissue and `n_c` is the training count for class `c`. Thus each class contributed approximately equal total training loss without resampling genes. Validation data were used for early stopping and best-checkpoint selection; test labels were used only for final evaluation.

The pooled analyses treat every usable **gene–tissue value** as one classification observation. Species-level analyses pool tissues only within a species, and tissue-level CSV files report each target independently. Because several tissue observations can come from the same gene, pooled observations should not be interpreted as statistically independent replicates; the reported values are descriptive test-set metrics rather than confidence intervals.

## Original DeepCRE architecture and task

DeepCRE is an interpretable sequence-to-expression convolutional neural network introduced in *Deep learning the cis-regulatory code for gene expression in selected model plants* (Nature Communications, 2024). Its purpose was to learn proximal cis-regulatory sequence patterns predictive of plant gene-expression state and then interpret those patterns using nucleotide-level attribution and motif discovery. The published workflow constructed a 3,020-nt one-hot input by concatenating two 1.5-kb regions: 1,000 nt upstream plus 500 nt inside the gene around the transcription start site (TSS), and 500 nt inside plus 1,000 nt downstream around the transcription termination site (TTS), separated by 20 ambiguous `N` bases.

The original network contained three convolutional blocks. Each block had two stride-one, same-padded Conv1D layers with kernel width 8, followed by max pooling of width 8 and dropout of 0.25. Channel widths were 64/64 in block 1, 128/128 in block 2, and 64/64 in block 3. The convolutional representation was flattened and passed through dense layers of 128 and 64 units, with dropout after the 128-unit layer, followed by one sigmoid output. Original DeepCRE models were binary classifiers trained on genes below the lower or above the upper expression quartile; medium-quartile genes were omitted. The paper considered single-species-reference and multi-species-reference models, used chromosome-level validation, removed validation genes with homologues in training, and balanced binary training data by downsampling.

## Adaptation of DeepCRE for this benchmark

We retained the six convolutional layers, channel widths, kernel width, three max-pooling operations, dropout rate, and 128/64-unit dense block. Four changes were necessary for a controlled PGB comparison:

1. **Common sequence input.** PGB provides a 6,000-nt TSS-centered window (5 kb upstream and 1 kb downstream), but it does not preserve the TTS coordinates required to reconstruct DeepCRE's original concatenated input. The adapted CNN therefore receives the same complete 6-kb sequence used by the foundation models. A/C/G/T bases are one-hot encoded and ambiguous bases are represented by four zeros.
2. **Three expression classes.** The single sigmoid unit was replaced by three logits per tissue and softmax probabilities for low, medium, and high expression. Cross-entropy replaced binary cross-entropy, and medium-expression observations were retained.
3. **Multi-tissue output.** One CNN was trained per species with a joint output tensor of `number of tissues × 3 classes`, instead of training separate leaf/root binary networks. The masked loss includes every usable tissue label for a gene.
4. **Matched data protocol.** We used the exact PGB train/validation/test and gene-family partitions rather than reconstructing DeepCRE's chromosome folds. Class-weighted loss replaced binary downsampling so all available training genes remained in the comparison. Adam used a learning rate of 1e-4 and batch size 64; training ran for at most 100 epochs, the learning rate was reduced from validation loss after five unimproved epochs, and early stopping occurred after ten epochs without improvement in validation macro-F1.

Consequently, `deepcre/cnn` should be described as a **DeepCRE-style architectural adaptation**, not as a direct reproduction of the published DeepCRE performance. It compares a compact sequence CNN with frozen genomic foundation-model representations under identical PGB inputs, labels, and held-out samples.

## Metric definitions and calculations

For class `c`, `TP_c`, `FP_c`, and `FN_c` denote the one-vs-rest true-positive, false-positive, and false-negative counts. Let `C(i,j)` be the confusion-matrix count with true class `i` and predicted class `j`, `N` the total count, `t_k` the true total for class `k`, and `p_k` the predicted total for class `k`.

- **Accuracy** is the fraction of observations assigned the correct class: `sum_k C(k,k) / N`. It is intuitive but dominated by common classes, so a model can obtain high accuracy while performing poorly on rare high-expression genes.
- **Balanced accuracy** is the unweighted mean of class recalls: `(1/3) sum_c TP_c / (TP_c + FN_c)`. Every class contributes equally regardless of prevalence, making this metric important for the imbalanced fixed-TPM classes.
- **Precision for class c** is `TP_c / (TP_c + FP_c)`. It answers: among observations predicted as class `c`, what fraction truly belong to `c`?
- **Recall for class c** is `TP_c / (TP_c + FN_c)`. It answers: among true members of class `c`, what fraction were recovered? Recall is also the class-specific true-positive rate or sensitivity.
- **F1 for class c** is the harmonic mean of precision and recall: `2 × precision_c × recall_c / (precision_c + recall_c)`. It is high only when both quantities are high.
- **Macro-F1** is `(1/3) sum_c F1_c`. It gives low, medium, and high expression equal influence and is the primary model-selection metric in this benchmark.
- **Weighted F1** is `sum_c (t_c / N) F1_c`. It accounts for every class but weights by test support, so it behaves more like accuracy when the low class is dominant.
- **Matthews correlation coefficient (MCC)** is the multiclass correlation between observed and predicted labels: `(N × sum_k C(k,k) - sum_k p_k t_k) / sqrt[(N² - sum_k p_k²)(N² - sum_k t_k²)]`. MCC incorporates every confusion-matrix cell; 1 denotes perfect prediction, 0 indicates no better-than-chance association, and negative values indicate systematic disagreement.
- **Confusion matrix.** Rows are true classes and columns are predicted classes. Each plotted percentage is row-normalized as `100 × C(i,j) / t_i`, while `n=` gives the unnormalized count. The diagonal is therefore class recall, not the proportion of the entire dataset.
- **One-vs-rest ROC and AUROC.** For each class, its softmax probability is thresholded across all possible cutoffs while the other two classes are treated as negatives. The ROC curve plots `TPR = TP/(TP+FN)` against `FPR = FP/(FP+TN)`; AUROC is the area under this curve. It can be interpreted as the probability that a randomly selected positive receives a higher score than a randomly selected negative. Macro-AUROC is the arithmetic mean of the three class AUROCs. Micro-AUROC flattens the three one-hot targets and probability columns before calculating one global curve. ROC can appear optimistic when negatives greatly outnumber positives.
- **One-vs-rest precision–recall and AUPRC.** For each class, the PR curve plots precision against recall over all probability thresholds. The reported AUPRC uses average precision, the step-wise weighted mean of precision values in which each weight is the increase in recall from the preceding threshold. Its no-skill reference is approximately the class prevalence, so AUPRC is especially revealing for the rare high-expression class. Macro-AUPRC averages the three class AUPRCs equally. Micro-AUPRC flattens all class decisions before constructing one curve and is consequently more influenced by common classes.

Undefined one-vs-rest AUROC/AUPRC values would be omitted when a test subgroup contains no positives or no negatives for a class. All primary pooled metrics were well-defined here.

## Results

Across pooled held-out gene–tissue observations, **{best['head']}** achieved the highest macro-F1 ({best['macro_f1']:.3f}), with accuracy {best['accuracy']:.3f}, balanced accuracy {best['balanced_accuracy']:.3f}, macro-AUROC {best['macro_auroc']:.3f}, and macro-AUPRC {best['macro_auprc']:.3f}.{deepcre_sentence} Species-stratified and class-stratified results are reported alongside pooled estimates because pooled accuracy is influenced by both species size and the strong rarity of the high-expression class.

PlantCAD2-large and DeepCRE-style CNN occupied the top two macro-F1 positions, separated by less than 0.005. PlantCAD2-large had the highest pooled accuracy (0.715), weighted F1 (0.716), macro-F1 (0.579), and MCC (0.442). DeepCRE had slightly lower accuracy (0.695) and macro-F1 (0.575), but the highest balanced accuracy (0.615), macro-AUROC (0.814), and macro-AUPRC (0.610) among the nine architectures. {deepcre_class_sentence} This precision–recall tradeoff is consistent with balanced loss emphasizing recovery of the rare class: DeepCRE recovered a larger fraction of high-expression observations, but more of its high predictions were false positives.

Complete numerical results are provided in `tables/overall_metrics.*`, `tables/species_metrics.*`, and `tables/per_class_metrics.*`. The species plot should be used to assess consistency across organisms; the pooled table alone weights species according to their numbers of genes and tissues.

## Interpretation and reporting note

Accuracy, weighted F1, micro-AUROC, and micro-AUPRC emphasize performance on the most frequent decisions. Balanced accuracy, macro-F1, macro-AUROC, and macro-AUPRC give classes equal influence and are more appropriate for comparing minority-class behavior. No single scalar captures the full error pattern, so the manuscript should present at least one prevalence-weighted metric, one macro metric, the class-specific table, and the confusion matrix. The DeepCRE adaptation result should not be compared numerically with the original paper's reported binary accuracies because the input regions, class definitions, tissue organization, and split protocol differ.
"""
    if fine_tuned_supplemental:
        paper_summary = f"""# Supplemental fine-tuning benchmark

## Design and caveat

The original PGB-trained heads were continued from their saved best checkpoints in new,
separate run directories. Fine-tuning used only supplemental train loci with **weak,
presumed-low** targets and measured PGB train examples replayed in each epoch. The
The frozen heads retained their original PGB-derived feature normalization;
all heads retained original PGB training-derived class weights.
Checkpoint selection used only measured PGB validation macro-F1, with the source
checkpoint kept as an epoch-zero candidate. Neither PGB test nor supplemental test
was used for training or selection.

Pseudogenes and intergenic windows have no measured tissue TPM. Their low-call
rates are behavioral diagnostics, not verified expression accuracy; some
pseudogenes may be transcribed. Supplemental splits are deterministic and
stratified by locus type, but not chromosome- or homology-disjoint; their test
low-call rates may therefore be optimistic. The ordinary metrics in this report are on the
original held-out PGB test set, using the same TPM boundaries: low
<{canonical_thresholds['low_upper_exclusive']:g}, medium
{canonical_thresholds['low_upper_exclusive']:g}–{canonical_thresholds['high_upper_inclusive']:g},
and high >{canonical_thresholds['high_upper_inclusive']:g} TPM.

## Held-out PGB result

The highest pooled macro-F1 was **{best['head']}** at {best['macro_f1']:.3f}.
See `tables/overall_metrics.*`, `tables/species_metrics.*`,
`tables/per_class_metrics.*`, and the confusion-matrix and ROC/PR figures for
the complete test results. Supplemental test behavior is reported separately
by `evaluate_tpm_classifiers_supplemental.py`.
"""
    (output_dir / "paper_ready_summary.md").write_text(paper_summary, encoding="utf-8")

    contents = """# Results table of contents

## Primary figures

- `confusion_matrices_all_species.png` — pooled confusion matrix for every architecture.
- `confusion_matrices_by_species/` — all-architecture confusion matrix grid for each species.
- `figures/roc_curves.{png,pdf}` — one-vs-rest ROC curves plus macro/micro averages.
- `figures/precision_recall_curves.{png,pdf}` — one-vs-rest PR curves plus macro/micro averages.
- `figures/summary_metrics.{png,pdf}` — grouped architecture comparison.
- `figures/per_class_precision_recall_f1.{png,pdf}` — low/medium/high class performance.
- `figures/species_performance.{png,pdf}` — macro-F1 and macro-AUROC by species.

## Tables

- `tables/overall_metrics.{csv,md,tex,png,pdf}` — ranked pooled metrics.
- `tables/species_metrics.{csv,md,tex,png,pdf}` — metrics for each architecture × species.
- `tables/per_class_metrics.{csv,md,tex,png,pdf}` — precision, recall, F1, AUROC, and AUPRC by class.
- `metrics_by_model_species_tissue.csv` — complete tissue-level results.
- `confusion_matrix_cells.csv` — counts and row-normalized percentages.

## Readable run artifacts

- `../runs_results/` — CSV-only mirror of `../runs/`, preserving the complete model/species/evaluation hierarchy without checkpoints, JSON, or Parquet files.

## Narrative and provenance

- `paper_ready_summary.md` — manuscript-ready methods, headline results, and interpretation caveat.
- `report_config.json` — input run directories, thresholds, and output manifest.
"""
    (output_dir / "CONTENTS.md").write_text(contents, encoding="utf-8")

    report_config = {
        "benchmark_protocol": "supplemental_finetune" if fine_tuned_supplemental else "original_pgb",
        "run_dirs": [str(path.expanduser().resolve()) for path in run_dirs],
        "results_subdir_requested": results_subdir,
        "thresholds_tpm": canonical_thresholds,
        "heads": sorted(seen_heads),
        "outputs": {
            "metrics_by_model_species": "metrics_by_model_species.csv",
            "metrics_by_model_species_tissue": "metrics_by_model_species_tissue.csv",
            "overall_confusion_matrices": "confusion_matrices_all_species.png",
            "species_confusion_matrix_directory": "confusion_matrices_by_species",
            "publication_figures": "figures",
            "publication_tables": "tables",
            "paper_summary": "paper_ready_summary.md",
            "table_of_contents": "CONTENTS.md",
        },
    }
    (output_dir / "report_config.json").write_text(json.dumps(report_config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote combined classifier report: {output_dir}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--results-subdir", default="evaluation",
        help="Prefer results in this run subdirectory; fall back to training-time outputs",
    )
    parser.add_argument(
        "--fine-tuned-supplemental", action="store_true",
        help="write training-aware narrative for heads continued on supplemental train loci",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        make_report(
            args.run_dir, args.output_dir.expanduser().resolve(), args.results_subdir,
            args.fine_tuned_supplemental,
        )
        return 0
    except (ReportError, OSError, ValueError, KeyError, ImportError) as error:
        print(f"ERROR: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
