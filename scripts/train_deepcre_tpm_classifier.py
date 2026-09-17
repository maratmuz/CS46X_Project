#!/usr/bin/env python3
"""Train/evaluate a DeepCRE-style CNN on the PGB three-class TPM task.

This is an architecture comparison, not an exact reproduction of the original
DeepCRE experiment.  It retains DeepCRE's three convolutional blocks (two
kernel-8 convolutions, max-pooling, and dropout per block) and dense 128/64
head, while adapting the input/output contract to PGB:

* the same 6,000-nt TSS-centered sequence used by all foundation models;
* one three-way softmax target per tissue;
* the existing PGB train/validation/test and gene-family splits;
* optional balanced cross-entropy computed from training labels only.

No frozen embeddings are loaded or recomputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import train_tpm_classifiers as common


DEFAULT_SWEEP = Path("/nfs/hpc/share/evo2_shared/tpm-classification-sweeps/tpm_balanced_v1")
DEFAULT_DATASET = Path("/nfs/hpc/share/evo2_shared/datasets/pgb_parquet_tpm_3class")
SPECIES = common.SPECIES
SPLITS = common.SPLITS
CLASS_NAMES = common.CLASS_NAMES


class SameConv1d(nn.Module):
    """Keras-compatible stride-1 SAME padding for an even-width kernel."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 8) -> None:
        super().__init__()
        total = kernel_size - 1
        self.left = total // 2
        self.right = total - self.left
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left, self.right)))


class DeepCREClassifier(nn.Module):
    """DeepCRE convolutional/dense architecture with tissue × class outputs."""

    def __init__(self, sequence_length: int, n_tissues: int, dropout: float = 0.25) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = ((4, 64), (64, 128), (128, 64))
        for input_channels, output_channels in channels:
            layers.extend(
                [
                    SameConv1d(input_channels, output_channels, 8),
                    nn.ReLU(),
                    SameConv1d(output_channels, output_channels, 8),
                    nn.ReLU(),
                    nn.MaxPool1d(8, stride=8, ceil_mode=True),
                    nn.Dropout(dropout),
                ]
            )
        self.convolutional = nn.Sequential(*layers)
        pooled_length = sequence_length
        for _ in range(3):
            pooled_length = math.ceil(pooled_length / 8)
        self.n_tissues = n_tissues
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled_length, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_tissues * len(CLASS_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(self.convolutional(x)).reshape(-1, self.n_tissues, len(CLASS_NAMES))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_sequence_split(
    dataset_root: Path, manifest: dict[str, object], species: str, split: str
) -> dict[str, object]:
    path = dataset_root / species / f"{split}.parquet"
    frame = pd.read_parquet(path, columns=["name", "sequence", "tpm_labels", "class_labels"])
    expected = manifest["species"][species]["splits"][split]
    names = frame["name"].astype(str).to_numpy()
    if len(frame) != int(expected["rows"]) or common.names_digest(names) != expected["name_sha256"]:
        raise common.ClassifierError(f"{species}/{split}: row identity differs from manifest")
    sequences = frame["sequence"].astype(str).str.upper().tolist()
    lengths = {len(value) for value in sequences}
    if len(lengths) != 1:
        raise common.ClassifierError(f"{species}/{split}: sequences have inconsistent lengths")
    classes = common.stack_vectors(frame["class_labels"], np.int64, f"{species}/{split}")
    tpm = common.stack_vectors(frame["tpm_labels"], np.float32, f"{species}/{split} TPM")
    return {
        "names": names,
        "sequences": sequences,
        "sequence_length": next(iter(lengths)),
        "y": torch.from_numpy(classes),
        "tpm": tpm,
    }


def one_hot_sequences(sequences: Sequence[str], device: str) -> torch.Tensor:
    if not sequences:
        raise common.ClassifierError("cannot encode an empty sequence batch")
    length = len(sequences[0])
    if any(len(value) != length for value in sequences):
        raise common.ClassifierError("sequence batch has inconsistent lengths")
    encoded_bytes = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8).reshape(-1, length)
    lookup = np.zeros((256, 4), dtype=np.float32)
    for index, nucleotide in enumerate(b"ACGT"):
        lookup[nucleotide, index] = 1.0
    one_hot = lookup[encoded_bytes]
    return torch.from_numpy(one_hot).permute(0, 2, 1).to(device, non_blocking=True)


def infer_logits(
    model: nn.Module, data: dict[str, object], device: str, batch_size: int
) -> torch.Tensor:
    model.eval()
    output = []
    sequences = data["sequences"]
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = one_hot_sequences(sequences[start : start + batch_size], device)
            output.append(model(batch).float().cpu())
    return torch.cat(output)


def save_history(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def selection_value(metrics: dict[str, object], loss: float, selection: str) -> float:
    return -loss if selection == "loss" else float(metrics["overall"][selection])


def capture_rng_state(generator: torch.Generator) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "permutation_generator": generator.get_state(),
    }


def restore_rng_state(state: dict[str, object], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["permutation_generator"])


def completed_species_outputs(run_dir: Path, species: str) -> bool:
    species_dir = run_dir / species
    return all(
        (species_dir / filename).is_file()
        for filename in (
            "best.pt",
            "training_history.csv",
            "test_metrics.json",
            "test_metrics_by_tissue.csv",
            "test_predictions.parquet",
        )
    )


def load_completed_species(
    run_dir: Path, species: str
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    species_dir = run_dir / species
    metrics = json.loads((species_dir / "test_metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(species_dir / "test_predictions.parquet")
    tissue_table = pd.read_csv(species_dir / "test_metrics_by_tissue.csv")
    return metrics, predictions, tissue_table


def train_species(
    args: argparse.Namespace, manifest: dict[str, object], species: str, run_dir: Path
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    splits = {
        split: load_sequence_split(args.dataset_root, manifest, species, split) for split in SPLITS
    }
    lengths = {int(data["sequence_length"]) for data in splits.values()}
    if len(lengths) != 1:
        raise common.ClassifierError(f"{species}: sequence length differs across splits")
    sequence_length = next(iter(lengths))
    n_tissues = splits["train"]["y"].shape[1]
    targets = manifest["species"][species]["targets"]
    species_seed = common.stable_seed(args.seed, species, "deepcre", "cnn")
    seed_everything(species_seed)
    model = DeepCREClassifier(sequence_length, n_tissues, args.dropout).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )
    class_weights = common.compute_class_weights(splits["train"]["y"], args.class_weighting)
    class_weights_device = class_weights.to(args.device) if class_weights is not None else None
    generator = torch.Generator(device="cpu").manual_seed(species_seed)
    train_y = splits["train"]["y"]
    train_sequences = splits["train"]["sequences"]
    checkpoint_path = run_dir / species / "best.pt"
    history: list[dict[str, object]] = []
    best_value, best_epoch = -float("inf"), 0
    start_epoch = 1
    resume_source = None
    last_checkpoint_path = run_dir / species / "last.pt"

    if args.resume and last_checkpoint_path.is_file():
        resume = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        history = list(resume["history"])
        best_value = float(resume["best_value"])
        best_epoch = int(resume["best_epoch"])
        start_epoch = int(resume["epoch"]) + 1
        restore_rng_state(resume["rng_state"], generator)
        resume_source = f"last epoch {start_epoch - 1}"
        print(f"  {species}: resuming exactly after epoch {start_epoch - 1}", flush=True)
    elif args.resume and checkpoint_path.is_file():
        # Backward-compatible recovery for jobs started before full last-epoch
        # checkpoints were implemented. Optimizer/RNG state did not exist, so
        # continue from the best model rather than discarding all training.
        resume = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["state_dict"])
        best_value = float(resume["selection_value"])
        best_epoch = int(resume["best_epoch"])
        start_epoch = best_epoch + 1
        for _ in range(best_epoch):
            torch.randperm(len(train_sequences), generator=generator)
        resume_source = f"legacy best epoch {best_epoch} (optimizer/RNG reinitialized)"
        print(
            f"  {species}: legacy partial run; continuing from best epoch {best_epoch}. "
            "Optimizer and CUDA RNG state were not saved by the old job.",
            flush=True,
        )

    if start_epoch > args.epochs:
        raise common.ClassifierError(
            f"{species}: resume epoch {start_epoch} exceeds configured epoch cap {args.epochs}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_sequences), generator=generator)
        loss_sum, value_count = 0.0, 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            sequence_batch = [train_sequences[index] for index in indices.tolist()]
            x = one_hot_sequences(sequence_batch, args.device)
            y = train_y[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = common.masked_cross_entropy(logits, y, class_weights_device)
            loss.backward()
            optimizer.step()
            usable = int((y >= 0).sum())
            loss_sum += float(loss.detach()) * usable
            value_count += usable
        train_loss = loss_sum / max(value_count, 1)

        validation_logits = infer_logits(model, splits["validation"], args.device, args.eval_batch_size)
        validation_y = splits["validation"]["y"]
        validation_loss = float(
            common.masked_cross_entropy(validation_logits, validation_y, class_weights).item()
        )
        validation_metrics, _ = common.score_outputs(validation_y, validation_logits, targets)
        scheduler.step(validation_loss)
        value = selection_value(validation_metrics, validation_loss, args.selection_metric)
        improved = value > best_value + args.min_delta
        if improved:
            best_value, best_epoch = value, epoch
            common.save_checkpoint(
                {
                    "state_dict": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                    "sequence_length": sequence_length,
                    "n_tissues": n_tissues,
                    "num_classes": len(CLASS_NAMES),
                    "dropout": args.dropout,
                    "class_weights": class_weights,
                    "class_names": CLASS_NAMES,
                    "species": species,
                    "model": "deepcre",
                    "size": "cnn",
                    "seed": species_seed,
                    "thresholds_tpm": manifest["thresholds_tpm"],
                    "tissue_targets": targets,
                    "best_epoch": best_epoch,
                    "selection_metric": args.selection_metric,
                    "selection_value": best_value,
                    "architecture": "DeepCRE three-block CNN adapted to 6000-nt PGB TSS windows",
                },
                checkpoint_path,
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_metrics["overall"]["accuracy"],
                "validation_balanced_accuracy": validation_metrics["overall"]["balanced_accuracy"],
                "validation_macro_f1": validation_metrics["overall"]["macro_f1"],
                "validation_weighted_f1": validation_metrics["overall"]["weighted_f1"],
                "validation_mcc": validation_metrics["overall"]["mcc"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best": improved,
            }
        )
        common.save_checkpoint(
            {
                "state_dict": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_value": best_value,
                "history": history,
                "rng_state": capture_rng_state(generator),
                "species": species,
                "sequence_length": sequence_length,
                "n_tissues": n_tissues,
                "dropout": args.dropout,
            },
            last_checkpoint_path,
        )
        if epoch == 1 or epoch % args.log_every == 0 or improved:
            print(
                f"  {species} epoch={epoch} train_loss={train_loss:.5f} "
                f"val_loss={validation_loss:.5f} "
                f"val_macro_f1={validation_metrics['overall']['macro_f1']:.5f} "
                f"best={best_value:.5f}@{best_epoch}",
                flush=True,
            )
        if epoch - best_epoch >= args.patience:
            print(f"  {species}: early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_logits = infer_logits(model, splits["test"], args.device, args.eval_batch_size)
    metrics, probabilities = common.score_outputs(splits["test"]["y"], test_logits, targets)
    metrics.update(
        {
            "species": species,
            "model": "deepcre",
            "size": "cnn",
            "best_epoch": best_epoch,
            "stopped_epoch": epoch,
            "class_weighting": args.class_weighting,
            "class_weights_training_split_only": class_weights.tolist() if class_weights is not None else None,
            "architecture_adaptation": checkpoint["architecture"],
            "resume_source": resume_source,
        }
    )
    species_dir = run_dir / species
    save_history(history, species_dir / "training_history.csv")
    common.write_json_atomic(metrics, species_dir / "test_metrics.json")
    predictions = common.prediction_frame(species, splits["test"], test_logits, probabilities, targets)
    common.write_parquet_atomic(predictions, species_dir / "test_predictions.parquet")
    tissue_table = common.flatten_tissue_metrics(species, metrics)
    tissue_table.to_csv(species_dir / "test_metrics_by_tissue.csv", index=False)
    print(
        f"  {species}: test accuracy={metrics['overall']['accuracy']:.4f} "
        f"balanced_accuracy={metrics['overall']['balanced_accuracy']:.4f} "
        f"macro_F1={metrics['overall']['macro_f1']:.4f} MCC={metrics['overall']['mcc']:.4f}",
        flush=True,
    )
    return metrics, predictions, tissue_table


def train_command(args: argparse.Namespace) -> int:
    args.dataset_root = args.dataset_root.expanduser().resolve()
    manifest = common.load_manifest(args.dataset_root)
    common.print_distribution_before_training(args.dataset_root, args.species, args.small_class_min)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise common.ClassifierError("CUDA was requested but is unavailable; pass --device cpu")
    run_dir = args.run_dir.expanduser().resolve()
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "train",
        "dataset_root": str(args.dataset_root),
        "model": "deepcre",
        "size": "cnn",
        "display_name": "DeepCRE-style CNN",
        "species": args.species,
        "seed": args.seed,
        "sequence_input": "PGB 6000-nt TSS window (5 kb upstream + 1 kb downstream)",
        "architecture_source": "DeepCRE/model/utils.py and DeepCRE.pdf",
        "adaptation": (
            "Original six Conv1D/three MaxPool/dense architecture; input changed from the original "
            "3020-nt TSS+TTS concatenation to PGB's 6000-nt TSS window; sigmoid binary output changed "
            "to independent tissue-wise three-class logits."
        ),
        "kernel_size": 8,
        "convolution_channels": [64, 64, 128, 128, 64, 64],
        "pool_size": 8,
        "dense_units": [128, 64],
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
        "selection_metric": args.selection_metric,
        "class_weighting": args.class_weighting,
        "class_weights": "computed independently per tissue from training split only",
        "validation_use": "early stopping and best-checkpoint selection only",
        "test_use": "held-out final evaluation only",
        "embeddings": "not used",
    }
    nonempty = run_dir.exists() and any(run_dir.iterdir())
    if nonempty:
        if not args.resume:
            raise common.ClassifierError(f"run directory is nonempty and resume is disabled: {run_dir}")
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            raise common.ClassifierError(f"cannot resume without run configuration: {config_path}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = (
            "dataset_root", "model", "size", "species", "seed", "dropout", "batch_size",
            "learning_rate", "weight_decay", "lr_patience", "lr_factor", "min_lr",
            "selection_metric", "class_weighting",
        )
        changed = [key for key in comparable if existing.get(key) != config.get(key)]
        if changed:
            raise common.ClassifierError(
                f"resume configuration differs for {changed}; use the original settings or a new run directory"
            )
        print(f"Resuming existing DeepCRE run: {run_dir}", flush=True)
    else:
        common.write_json_atomic(config, run_dir / "run_config.json")
    metrics_by_species: dict[str, dict[str, object]] = {}
    predictions, tissue_tables = [], []
    for species in args.species:
        if args.resume and completed_species_outputs(run_dir, species):
            print(f"\nSkipping completed DeepCRE species: {species}", flush=True)
            metrics, prediction, tissue_table = load_completed_species(run_dir, species)
        else:
            print(f"\nTraining DeepCRE-style CNN for {species}", flush=True)
            metrics, prediction, tissue_table = train_species(args, manifest, species, run_dir)
        metrics_by_species[species] = metrics
        predictions.append(prediction)
        tissue_tables.append(tissue_table)
    common.aggregate_results("deepcre", "cnn", metrics_by_species, predictions, tissue_tables, run_dir)
    print(f"\nCompleted DeepCRE-style run: {run_dir}")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"])
    manifest = common.load_manifest(dataset_root)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise common.ClassifierError("CUDA was requested but is unavailable; pass --device cpu")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "evaluation"
    metrics_by_species: dict[str, dict[str, object]] = {}
    predictions_all, tissue_tables = [], []
    for species in config["species"]:
        checkpoint = torch.load(run_dir / species / "best.pt", map_location="cpu", weights_only=False)
        data = load_sequence_split(dataset_root, manifest, species, "test")
        model = DeepCREClassifier(
            checkpoint["sequence_length"], checkpoint["n_tissues"], checkpoint["dropout"]
        ).to(args.device)
        model.load_state_dict(checkpoint["state_dict"])
        logits = infer_logits(model, data, args.device, args.eval_batch_size)
        targets = manifest["species"][species]["targets"]
        metrics, probabilities = common.score_outputs(data["y"], logits, targets)
        metrics.update({"species": species, "model": "deepcre", "size": "cnn"})
        species_dir = output_dir / species
        common.write_json_atomic(metrics, species_dir / "test_metrics.json")
        prediction = common.prediction_frame(species, data, logits, probabilities, targets)
        common.write_parquet_atomic(prediction, species_dir / "test_predictions.parquet")
        tissue_table = common.flatten_tissue_metrics(species, metrics)
        tissue_table.to_csv(species_dir / "test_metrics_by_tissue.csv", index=False)
        metrics_by_species[species] = metrics
        predictions_all.append(prediction)
        tissue_tables.append(tissue_table)
    common.aggregate_results(
        "deepcre", "cnn", metrics_by_species, predictions_all, tissue_tables, output_dir
    )
    print(f"Wrote held-out DeepCRE-style evaluation: {output_dir}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    train.add_argument("--run-dir", type=Path, default=DEFAULT_SWEEP / "runs" / "deepcre_cnn")
    train.add_argument("--species", nargs="+", choices=SPECIES, default=list(SPECIES))
    train.add_argument("--class-weighting", choices=("none", "balanced"), default="balanced")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--dropout", type=float, default=0.25)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--eval-batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--lr-patience", type=int, default=5)
    train.add_argument("--lr-factor", type=float, default=0.1)
    train.add_argument("--min-lr", type=float, default=1e-7)
    train.add_argument("--min-delta", type=float, default=1e-5)
    train.add_argument(
        "--selection-metric", choices=("macro_f1", "balanced_accuracy", "loss"), default="macro_f1"
    )
    train.add_argument("--small-class-min", type=int, default=20)
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=1)
    train.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="resume last.pt, fall back to best.pt for legacy partial runs, and skip completed species",
    )
    train.set_defaults(func=train_command)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=None)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--eval-batch-size", type=int, default=128)
    evaluate.set_defaults(func=evaluate_command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        return args.func(args)
    except (common.ClassifierError, OSError, ValueError, KeyError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
