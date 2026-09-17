#!/usr/bin/env python3
"""Train or evaluate tissue-specific 3-way heads on frozen PGB embeddings.

This script never invokes a genomic foundation model. It loads the existing
``X`` tensors under frozen-embeddings and replaces their cached regression
targets with row-aligned class labels from ``pgb_parquet_tpm_3class``.

Examples:
  python scripts/train_tpm_classifiers.py train --model evo2 --size 7b
  python scripts/train_tpm_classifiers.py evaluate --run-dir PATH_TO_RUN
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

# PyTorch requires this to be set before CUDA/cuBLAS is initialized when
# deterministic algorithms are requested.  Keeping the default here also
# covers direct invocations that do not go through the sweep shell script.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


SHARED_DIR = Path(os.environ.get("SHARED_DIR", "/nfs/hpc/share/evo2_shared"))
DEFAULT_DATASET = SHARED_DIR / "datasets" / "pgb_parquet_tpm_3class"
DEFAULT_EMBED_ROOT = SHARED_DIR / "frozen-embeddings"
DEFAULT_OUTPUT_ROOT = SHARED_DIR / "tpm-classification-heads"
SPECIES = (
    "arabidopsis_thaliana",
    "glycine_max",
    "oryza_sativa",
    "solanum_lycopersicum",
    "zea_mays",
)
SPLITS = ("train", "validation", "test")
CLASS_NAMES = ("low", "medium", "high")
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = -1


class ClassifierError(RuntimeError):
    pass


class TissueClassifierHead(nn.Module):
    """Regression-head-sized MLP producing three logits for every tissue."""

    def __init__(
        self, in_dim: int, n_tissues: int, hidden_dim: int = 1024, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.n_tissues = n_tissues
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tissues * NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1, self.n_tissues, NUM_CLASSES)


def stable_seed(seed: int, *parts: str) -> int:
    value = hashlib.sha256("\0".join((str(seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(value[:4], "big")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def names_digest(names: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in names:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stack_vectors(series: pd.Series, dtype: np.dtype, description: str) -> np.ndarray:
    try:
        rows = [np.asarray(value, dtype=dtype) for value in series]
        if not rows:
            raise ClassifierError(f"{description}: split is empty")
        shapes = {row.shape for row in rows}
        if len(shapes) != 1 or rows[0].ndim != 1:
            raise ClassifierError(f"{description}: target vectors have inconsistent shapes")
        return np.stack(rows)
    except (TypeError, ValueError) as error:
        raise ClassifierError(f"{description}: invalid target vectors: {error}") from error


def load_manifest(dataset_root: Path) -> dict[str, object]:
    path = dataset_root / "manifest.json"
    if not path.is_file():
        raise ClassifierError(f"classification manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "pgb-tpm-tissue-classification-v1":
        raise ClassifierError(f"unsupported dataset format in {path}")
    return manifest


def load_label_split(
    dataset_root: Path, manifest: dict[str, object], species: str, split: str
) -> dict[str, object]:
    path = dataset_root / species / f"{split}.parquet"
    if not path.is_file():
        raise ClassifierError(f"classification split is missing: {path}")
    frame = pd.read_parquet(path, columns=["name", "tpm_labels", "class_labels"])
    classes = stack_vectors(frame["class_labels"], np.int64, f"{species}/{split}")
    tpm = stack_vectors(frame["tpm_labels"], np.float32, f"{species}/{split} TPM")
    if classes.shape != tpm.shape:
        raise ClassifierError(f"{species}/{split}: TPM and class target shapes differ")
    if np.any((classes < IGNORE_INDEX) | (classes >= NUM_CLASSES)):
        raise ClassifierError(f"{species}/{split}: class labels are outside -1,0,1,2")
    expected = manifest["species"][species]["splits"][split]
    if len(frame) != int(expected["rows"]):
        raise ClassifierError(f"{species}/{split}: row count differs from manifest")
    if names_digest(frame["name"]) != expected["name_sha256"]:
        raise ClassifierError(f"{species}/{split}: gene IDs or row order differ from manifest")
    return {
        "names": frame["name"].astype(str).to_numpy(),
        "tpm": tpm,
        "y": torch.from_numpy(classes),
    }


def resolve_embedding_path(
    embed_root: Path,
    model: str,
    size: str,
    species: str,
    split: str,
    layer: int | None,
) -> tuple[Path, int | None]:
    directory = embed_root / model / size / species
    if layer is not None:
        path = directory / f"{split}-L{layer}.pt"
        if not path.is_file():
            raise ClassifierError(f"requested embedding cache is missing: {path}")
        return path, layer
    untagged = directory / f"{split}.pt"
    if untagged.is_file():
        return untagged, None
    tagged = sorted(directory.glob(f"{split}-L*.pt"))
    if len(tagged) != 1:
        choices = ", ".join(path.name for path in tagged) or "none"
        raise ClassifierError(
            f"cannot infer layer for {model}/{size}/{species}/{split}; candidates: {choices}. "
            "Pass --layer."
        )
    stem = tagged[0].stem
    try:
        resolved_layer = int(stem.rsplit("-L", 1)[1])
    except ValueError as error:
        raise ClassifierError(f"invalid layer-tagged cache name: {tagged[0]}") from error
    return tagged[0], resolved_layer


def load_embedding_split(
    embed_root: Path,
    model: str,
    size: str,
    species: str,
    split: str,
    layer: int | None,
    expected_rows: int,
) -> tuple[torch.Tensor, Path, int | None]:
    path, resolved_layer = resolve_embedding_path(embed_root, model, size, species, split, layer)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "X" not in data:
        raise ClassifierError(f"embedding cache does not contain X: {path}")
    X = data["X"].float()
    if X.ndim != 2:
        raise ClassifierError(f"embedding cache X must be rank 2: {path}")
    if len(X) != expected_rows:
        raise ClassifierError(
            f"row mismatch for {species}/{split}: labels={expected_rows}, embeddings={len(X)} ({path})"
        )
    return X, path, resolved_layer


def compute_class_weights(train_y: torch.Tensor, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    weights = torch.zeros((train_y.shape[1], NUM_CLASSES), dtype=torch.float32)
    for tissue in range(train_y.shape[1]):
        target = train_y[:, tissue]
        valid = target >= 0
        n = int(valid.sum())
        for class_index in range(NUM_CLASSES):
            count = int((target[valid] == class_index).sum())
            weights[tissue, class_index] = 0.0 if count == 0 else n / (NUM_CLASSES * count)
    return weights


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = target >= 0
    if not bool(valid.any()):
        raise ClassifierError("batch contains no usable labels")
    per_value = F.cross_entropy(logits[valid], target[valid], reduction="none")
    if class_weights is None:
        return per_value.mean()
    tissue_grid = torch.arange(target.shape[1], device=target.device).expand(target.shape[0], -1)
    value_weights = class_weights[tissue_grid[valid], target[valid]]
    denominator = value_weights.sum()
    if float(denominator) <= 0:
        raise ClassifierError("class weights have zero mass for every usable batch value")
    return (per_value * value_weights).sum() / denominator


def infer_logits(
    model: nn.Module,
    X: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    outputs = []
    mean_d, std_d = mean.to(device), std.to(device)
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = X[start : start + batch_size].to(device)
            outputs.append(model((xb - mean_d) / std_d).float().cpu())
    return torch.cat(outputs)


def optional_mean(values: Sequence[float | None]) -> float | None:
    usable = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(usable)) if usable else None


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    predicted = probabilities.argmax(axis=1)
    labels = list(range(NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=labels, zero_division=0
    )
    per_class: dict[str, object] = {}
    aurocs: list[float | None] = []
    auprcs: list[float | None] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary = (y_true == class_index).astype(np.int8)
        if binary.min() == binary.max():
            auroc = None
            auprc = None
        else:
            auroc = float(roc_auc_score(binary, probabilities[:, class_index]))
            auprc = float(average_precision_score(binary, probabilities[:, class_index]))
        aurocs.append(auroc)
        auprcs.append(auprc)
        per_class[class_name] = {
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(f1[class_index]),
            "support": int(support[class_index]),
            "auroc_ovr": auroc,
            "auprc_ovr": auprc,
        }
    total = int(support.sum())
    weighted_f1 = float(np.dot(f1, support) / total) if total else None
    return {
        "n": total,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": weighted_f1,
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=labels).astype(int).tolist(),
        "confusion_matrix_labels": list(CLASS_NAMES),
        "auroc_ovr_macro_defined_classes": optional_mean(aurocs),
        "auprc_ovr_macro_defined_classes": optional_mean(auprcs),
    }


def score_outputs(
    y: torch.Tensor, logits: torch.Tensor, targets: Sequence[dict[str, object]]
) -> tuple[dict[str, object], np.ndarray]:
    y_np = y.numpy()
    probabilities = torch.softmax(logits, dim=-1).numpy()
    valid = y_np >= 0
    if not valid.any():
        raise ClassifierError("evaluation split has no usable labels")
    overall = classification_metrics(y_np[valid], probabilities[valid])
    per_tissue = []
    for index, target in enumerate(targets):
        tissue_valid = valid[:, index]
        if tissue_valid.any():
            metrics = classification_metrics(y_np[tissue_valid, index], probabilities[tissue_valid, index])
        else:
            metrics = {"n": 0, "error": "no usable TPM values"}
        metrics.update(
            {
                "tissue_index": index,
                "target_name": target["target_name"],
                "tissue": target["tissue"],
                "source_run_ids": target["source_run_ids"],
            }
        )
        per_tissue.append(metrics)
    return {"overall": overall, "per_tissue": per_tissue}, probabilities


def write_json_atomic(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def write_history(history: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prediction_frame(
    species: str,
    data: dict[str, object],
    logits: torch.Tensor,
    probabilities: np.ndarray,
    targets: Sequence[dict[str, object]],
) -> pd.DataFrame:
    y = data["y"].numpy()
    tpm = data["tpm"]
    n, n_tissues = y.shape
    predicted = probabilities.argmax(axis=2)
    true_names = np.full(y.shape, "missing_or_invalid", dtype=object)
    predicted_names = np.empty(y.shape, dtype=object)
    for index, name in enumerate(CLASS_NAMES):
        true_names[y == index] = name
        predicted_names[predicted == index] = name
    return pd.DataFrame(
        {
            "species": species,
            "gene_id": np.repeat(data["names"], n_tissues),
            "embedding_row": np.repeat(np.arange(n, dtype=np.int64), n_tissues),
            "tissue_index": np.tile(np.arange(n_tissues, dtype=np.int16), n),
            "target_name": np.tile([target["target_name"] for target in targets], n),
            "tissue": np.tile([target["tissue"] for target in targets], n),
            "tpm": tpm.reshape(-1),
            "valid_tpm": (y.reshape(-1) >= 0),
            "true_class": y.reshape(-1),
            "true_class_name": true_names.reshape(-1),
            "predicted_class": predicted.reshape(-1),
            "predicted_class_name": predicted_names.reshape(-1),
            "logit_low": logits[:, :, 0].numpy().reshape(-1),
            "logit_medium": logits[:, :, 1].numpy().reshape(-1),
            "logit_high": logits[:, :, 2].numpy().reshape(-1),
            "probability_low": probabilities[:, :, 0].reshape(-1),
            "probability_medium": probabilities[:, :, 1].reshape(-1),
            "probability_high": probabilities[:, :, 2].reshape(-1),
        }
    )


def flatten_tissue_metrics(species: str, metrics: dict[str, object]) -> pd.DataFrame:
    rows = []
    for tissue in metrics["per_tissue"]:
        row = {
            "species": species,
            "tissue_index": tissue["tissue_index"],
            "target_name": tissue["target_name"],
            "tissue": tissue["tissue"],
            "n": tissue["n"],
        }
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc"):
            row[key] = tissue.get(key)
        for class_name in CLASS_NAMES:
            values = tissue.get("per_class", {}).get(class_name, {})
            for key in ("precision", "recall", "f1", "support", "auroc_ovr", "auprc_ovr"):
                row[f"{class_name}_{key}"] = values.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def print_distribution_before_training(dataset_root: Path, species: Sequence[str], minimum: int) -> None:
    path = dataset_root / "class_distribution_by_tissue_split.csv"
    if not path.is_file():
        raise ClassifierError(f"class distribution summary is missing: {path}")
    detail = pd.read_csv(path)
    detail = detail[detail["species"].isin(species)].copy()
    columns = [
        "species", "tissue_index", "target_name", "split", "usable", "missing_invalid",
        "low_count", "low_percent", "medium_count", "medium_percent", "high_count", "high_percent",
    ]
    print("\nCLASS DISTRIBUTION — PRINTED BEFORE ANY TRAINING", flush=True)
    print(detail[columns].to_string(index=False, float_format=lambda value: f"{value:.2f}"), flush=True)
    warnings = []
    for row in detail.itertuples(index=False):
        for class_name in CLASS_NAMES:
            count = int(getattr(row, f"{class_name}_count"))
            if count == 0:
                warnings.append(f"NO {class_name.upper()}: {row.species}/{row.target_name}/{row.split}")
            elif count < minimum:
                warnings.append(
                    f"SMALL {class_name} ({count} < {minimum}): "
                    f"{row.species}/{row.target_name}/{row.split}"
                )
    if warnings:
        print("\n" + "!" * 96, file=sys.stderr, flush=True)
        print(f"PROMINENT CLASS-IMBALANCE WARNING ({len(warnings)} cells)", file=sys.stderr, flush=True)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr, flush=True)
        print("!" * 96 + "\n", file=sys.stderr, flush=True)


def validation_selection_value(metrics: dict[str, object], loss: float, selection: str) -> float:
    if selection == "loss":
        return -loss
    return float(metrics["overall"][selection])


def train_one_species(
    args: argparse.Namespace,
    manifest: dict[str, object],
    species: str,
    run_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    dataset_root = args.dataset_root.expanduser().resolve()
    embed_root = args.embed_root.expanduser().resolve()
    labels = {split: load_label_split(dataset_root, manifest, species, split) for split in SPLITS}
    embeddings: dict[str, torch.Tensor] = {}
    paths: dict[str, str] = {}
    resolved_layers = set()
    for split in SPLITS:
        X, path, resolved = load_embedding_split(
            embed_root, args.model, args.size, species, split, args.layer, len(labels[split]["y"])
        )
        embeddings[split] = X
        paths[split] = str(path)
        resolved_layers.add(resolved)
    if len(resolved_layers) != 1:
        raise ClassifierError(f"{species}: embedding layer differs across splits: {resolved_layers}")
    resolved_layer = next(iter(resolved_layers))

    n_tissues = labels["train"]["y"].shape[1]
    targets = manifest["species"][species]["targets"]
    if len(targets) != n_tissues:
        raise ClassifierError(f"{species}: tissue metadata and class targets differ")
    if any(labels[split]["y"].shape[1] != n_tissues for split in SPLITS):
        raise ClassifierError(f"{species}: tissue target count differs across splits")

    species_seed = stable_seed(args.seed, species, args.model, args.size)
    seed_everything(species_seed)
    train_X = embeddings["train"]
    feature_mean = train_X.mean(dim=0, keepdim=True)
    feature_std = train_X.std(dim=0, keepdim=True).clamp_min(1e-6)
    class_weights = compute_class_weights(labels["train"]["y"], args.class_weighting)
    class_weights_device = class_weights.to(args.device) if class_weights is not None else None

    model = TissueClassifierHead(
        train_X.shape[1], n_tissues, hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )
    mean_d, std_d = feature_mean.to(args.device), feature_std.to(args.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(species_seed)
    train_y = labels["train"]["y"]
    n = len(train_X)
    best_value = -float("inf")
    best_epoch = 0
    history: list[dict[str, object]] = []
    checkpoint_path = run_dir / species / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        weight_sum = 0
        for start in range(0, n, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            xb = train_X[indices].to(args.device)
            yb = train_y[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model((xb - mean_d) / std_d)
            loss = masked_cross_entropy(logits, yb, class_weights_device)
            loss.backward()
            optimizer.step()
            batch_values = int((yb >= 0).sum())
            loss_sum += float(loss.detach()) * batch_values
            weight_sum += batch_values
        train_loss = loss_sum / max(weight_sum, 1)

        val_logits = infer_logits(
            model, embeddings["validation"], feature_mean, feature_std, args.device, args.eval_batch_size
        )
        val_y = labels["validation"]["y"]
        val_loss = float(masked_cross_entropy(val_logits, val_y, class_weights).item())
        val_metrics, _ = score_outputs(val_y, val_logits, targets)
        scheduler.step(val_loss)
        selection_value = validation_selection_value(val_metrics, val_loss, args.selection_metric)
        improved = selection_value > best_value + args.min_delta
        if improved:
            best_value = selection_value
            best_epoch = epoch
            checkpoint = {
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "in_dim": train_X.shape[1],
                "n_tissues": n_tissues,
                "num_classes": NUM_CLASSES,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "class_weights": class_weights,
                "class_names": CLASS_NAMES,
                "species": species,
                "model": args.model,
                "size": args.size,
                "layer": resolved_layer,
                "seed": species_seed,
                "thresholds_tpm": manifest["thresholds_tpm"],
                "tissue_targets": targets,
                "embedding_paths": paths,
                "best_epoch": best_epoch,
                "selection_metric": args.selection_metric,
                "selection_value": best_value,
            }
            save_checkpoint(checkpoint, checkpoint_path)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_accuracy": val_metrics["overall"]["accuracy"],
                "validation_balanced_accuracy": val_metrics["overall"]["balanced_accuracy"],
                "validation_macro_f1": val_metrics["overall"]["macro_f1"],
                "validation_weighted_f1": val_metrics["overall"]["weighted_f1"],
                "validation_mcc": val_metrics["overall"]["mcc"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )
        if epoch == 1 or epoch % args.log_every == 0 or improved:
            print(
                f"  {species} epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"val_macro_f1={val_metrics['overall']['macro_f1']:.5f} "
                f"best={best_value:.5f}@{best_epoch}",
                flush=True,
            )
        if epoch - best_epoch >= args.patience:
            print(f"  {species}: early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_logits = infer_logits(
        model, embeddings["test"], feature_mean, feature_std, args.device, args.eval_batch_size
    )
    test_metrics, probabilities = score_outputs(labels["test"]["y"], test_logits, targets)
    test_metrics.update(
        {
            "species": species,
            "model": args.model,
            "size": args.size,
            "layer": resolved_layer,
            "best_epoch": best_epoch,
            "stopped_epoch": epoch,
            "class_weighting": args.class_weighting,
            "class_weights_training_split_only": class_weights.tolist() if class_weights is not None else None,
        }
    )
    species_dir = run_dir / species
    write_history(history, species_dir / "training_history.csv")
    write_json_atomic(test_metrics, species_dir / "test_metrics.json")
    predictions = prediction_frame(species, labels["test"], test_logits, probabilities, targets)
    write_parquet_atomic(predictions, species_dir / "test_predictions.parquet")
    tissue_table = flatten_tissue_metrics(species, test_metrics)
    tissue_table.to_csv(species_dir / "test_metrics_by_tissue.csv", index=False)
    print(
        f"  {species}: test accuracy={test_metrics['overall']['accuracy']:.4f} "
        f"balanced_accuracy={test_metrics['overall']['balanced_accuracy']:.4f} "
        f"macro_F1={test_metrics['overall']['macro_f1']:.4f} MCC={test_metrics['overall']['mcc']:.4f}",
        flush=True,
    )
    return test_metrics, predictions, tissue_table


def aggregate_results(
    model: str,
    size: str,
    species_metrics: dict[str, dict[str, object]],
    prediction_frames: Sequence[pd.DataFrame],
    tissue_frames: Sequence[pd.DataFrame],
    output_dir: Path,
) -> dict[str, object]:
    predictions = pd.concat(prediction_frames, ignore_index=True)
    valid = predictions["valid_tpm"].to_numpy(dtype=bool)
    y = predictions.loc[valid, "true_class"].to_numpy(dtype=np.int64)
    probabilities = predictions.loc[
        valid, ["probability_low", "probability_medium", "probability_high"]
    ].to_numpy(dtype=np.float64)
    pooled = classification_metrics(y, probabilities)
    scalar_keys = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc")
    macro_species = {
        key: float(np.mean([metrics["overall"][key] for metrics in species_metrics.values()]))
        for key in scalar_keys
    }
    summary = {
        "model": model,
        "size": size,
        "species": species_metrics,
        "macro_mean_across_species": macro_species,
        "pooled_gene_tissue_values": pooled,
    }
    write_json_atomic(summary, output_dir / "summary.json")
    summary_rows = []
    for species, metrics in species_metrics.items():
        summary_rows.append({"species": species, **{key: metrics["overall"][key] for key in scalar_keys}})
    summary_rows.append({"species": "MACRO_SPECIES_MEAN", **macro_species})
    summary_rows.append({"species": "POOLED", **{key: pooled[key] for key in scalar_keys}})
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(tissue_frames, ignore_index=True).to_csv(output_dir / "test_metrics_by_species_tissue.csv", index=False)
    return summary


def train_command(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest = load_manifest(dataset_root)
    print_distribution_before_training(dataset_root, args.species, args.small_class_min)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ClassifierError("CUDA was requested but is unavailable; pass --device cpu for a smoke test")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir
        else (args.output_root.expanduser().resolve() / f"{args.model}_{args.size}_{stamp}")
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ClassifierError(f"run directory is nonempty: {run_dir}")
    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "train",
        "dataset_root": str(dataset_root),
        "embed_root": str(args.embed_root.expanduser().resolve()),
        "model": args.model,
        "size": args.size,
        "layer_requested": args.layer,
        "species": args.species,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr_patience": args.lr_patience,
        "lr_factor": args.lr_factor,
        "min_lr": args.min_lr,
        "min_delta": args.min_delta,
        "selection_metric": args.selection_metric,
        "class_weighting": args.class_weighting,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "normalization": "per-feature z-score from training split only",
        "class_weights": "computed independently per tissue from training split only",
        "validation_use": "early stopping and best-checkpoint selection only",
        "test_use": "held-out final evaluation only",
    }
    write_json_atomic(run_config, run_dir / "run_config.json")
    metrics_by_species = {}
    predictions = []
    tissue_tables = []
    for species in args.species:
        print(f"\nTraining {args.model}/{args.size} for {species}", flush=True)
        metrics, prediction, tissue_table = train_one_species(args, manifest, species, run_dir)
        metrics_by_species[species] = metrics
        predictions.append(prediction)
        tissue_tables.append(tissue_table)
    aggregate_results(args.model, args.size, metrics_by_species, predictions, tissue_tables, run_dir)
    print(f"\nCompleted run: {run_dir}")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise ClassifierError(f"run configuration is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"])
    embed_root = Path(config["embed_root"])
    manifest = load_manifest(dataset_root)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ClassifierError("CUDA was requested but is unavailable; pass --device cpu")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "evaluation"
    metrics_by_species = {}
    predictions_all = []
    tissue_tables = []
    for species in config["species"]:
        checkpoint_path = run_dir / species / "best.pt"
        if not checkpoint_path.is_file():
            raise ClassifierError(f"checkpoint is missing: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        data = load_label_split(dataset_root, manifest, species, "test")
        X, embedding_path, resolved_layer = load_embedding_split(
            embed_root,
            config["model"],
            config["size"],
            species,
            "test",
            checkpoint["layer"],
            len(data["y"]),
        )
        model = TissueClassifierHead(
            checkpoint["in_dim"], checkpoint["n_tissues"],
            hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"],
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        logits = infer_logits(
            model, X, checkpoint["feature_mean"], checkpoint["feature_std"],
            device, args.eval_batch_size,
        )
        targets = manifest["species"][species]["targets"]
        metrics, probabilities = score_outputs(data["y"], logits, targets)
        metrics.update(
            {
                "species": species,
                "model": config["model"],
                "size": config["size"],
                "layer": resolved_layer,
                "embedding_path": str(embedding_path),
            }
        )
        species_dir = output_dir / species
        write_json_atomic(metrics, species_dir / "test_metrics.json")
        predictions = prediction_frame(species, data, logits, probabilities, targets)
        write_parquet_atomic(predictions, species_dir / "test_predictions.parquet")
        tissue_table = flatten_tissue_metrics(species, metrics)
        tissue_table.to_csv(species_dir / "test_metrics_by_tissue.csv", index=False)
        metrics_by_species[species] = metrics
        predictions_all.append(predictions)
        tissue_tables.append(tissue_table)
    aggregate_results(
        config["model"], config["size"], metrics_by_species, predictions_all, tissue_tables, output_dir
    )
    print(f"Wrote held-out test evaluation: {output_dir}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train heads and evaluate the held-out test split")
    train.add_argument("--model", required=True, choices=("plantcad2", "evo2", "agront", "ntv3"))
    train.add_argument("--size", required=True)
    train.add_argument("--layer", type=int, default=None)
    train.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    train.add_argument("--embed-root", type=Path, default=DEFAULT_EMBED_ROOT)
    train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    train.add_argument("--run-dir", type=Path, default=None)
    train.add_argument("--species", nargs="+", choices=SPECIES, default=list(SPECIES))
    train.add_argument("--class-weighting", choices=("none", "balanced"), default="none")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--hidden-dim", type=int, default=1024)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--eval-batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--epochs", type=int, default=4000)
    train.add_argument("--patience", type=int, default=250)
    train.add_argument("--lr-patience", type=int, default=80)
    train.add_argument("--lr-factor", type=float, default=0.5)
    train.add_argument("--min-lr", type=float, default=1e-6)
    train.add_argument("--min-delta", type=float, default=1e-5)
    train.add_argument(
        "--selection-metric", choices=("macro_f1", "balanced_accuracy", "loss"), default="macro_f1"
    )
    train.add_argument("--small-class-min", type=int, default=20)
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=25)
    train.set_defaults(func=train_command)

    evaluate = subparsers.add_parser("evaluate", help="re-evaluate saved heads on held-out test splits")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=None)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--eval-batch-size", type=int, default=512)
    evaluate.set_defaults(func=evaluate_command)
    return parser.parse_args(argv)


def validate_hyperparameters(args: argparse.Namespace) -> None:
    if args.command != "train":
        return
    positive = {
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr_patience": args.lr_patience,
        "min_lr": args.min_lr,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ClassifierError(f"hyperparameters must be positive: {invalid}")
    if not 0 <= args.dropout < 1:
        raise ClassifierError("dropout must be in [0, 1)")
    if args.weight_decay < 0 or not 0 < args.lr_factor < 1 or args.min_delta < 0:
        raise ClassifierError("invalid weight-decay, lr-factor, or min-delta")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        validate_hyperparameters(args)
        return args.func(args)
    except (ClassifierError, OSError, ValueError, ImportError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
