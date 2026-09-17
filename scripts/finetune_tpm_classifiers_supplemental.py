#!/usr/bin/env python3
"""Continue saved TPM heads with weak supplemental low targets and PGB replay.

Only supplemental *train* rows and measured PGB *train* rows update weights.
Measured PGB validation selects the best checkpoint; neither test partition is
read here. Supplemental 'low' is an explicit weak assumption, not a TPM label.
Original checkpoints are read-only and never replaced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import train_deepcre_tpm_classifier as deepcre
import train_tpm_classifiers as common


LOCUS_SETS = ("pseudogenes", "intergenic")


def supplemental_path(root: Path, locus_set: str, species: str) -> Path:
    path = root / locus_set / species / "train.parquet"
    if not path.is_file():
        raise common.ClassifierError(f"supplemental train split is missing: {path}")
    return path


def validate_identifiers(
    supplemental_names: Sequence[str], pgb_names: Sequence[str], context: str
) -> None:
    if len(set(supplemental_names)) != len(supplemental_names):
        raise common.ClassifierError(f"{context}: duplicate supplemental locus identifiers")
    overlap = set(supplemental_names).intersection(pgb_names)
    if overlap:
        raise common.ClassifierError(
            f"{context}: supplemental/PGB identifiers overlap, for example {sorted(overlap)[:3]}"
        )


def load_supplemental(
    root: Path,
    embed_root: Path,
    model_name: str,
    size: str,
    species: str,
    checkpoint: dict[str, object],
    pgb_names: Sequence[str],
) -> tuple[torch.Tensor | list[str], dict[str, object]]:
    chunks: list[torch.Tensor | list[str]] = []
    paths: list[str] = []
    counts: dict[str, int] = {}
    all_names: list[str] = []
    for locus_set in LOCUS_SETS:
        path = supplemental_path(root, locus_set, species)
        columns = ["name", "sequence"] if model_name == "deepcre" else ["name"]
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) == 0:
            raise common.ClassifierError(f"empty supplemental training split: {path}")
        names = frame["name"].astype(str).tolist()
        all_names.extend(names)
        counts[locus_set] = len(names)
        paths.append(str(path))
        if model_name == "deepcre":
            sequences = frame["sequence"].astype(str).str.upper().tolist()
            if any(len(sequence) != checkpoint["sequence_length"] for sequence in sequences):
                raise common.ClassifierError(
                    f"{species}/{locus_set}: sequence length differs from DeepCRE checkpoint"
                )
            chunks.append(sequences)
        else:
            cache_path = embed_root / locus_set / model_name / size / species
            layer = checkpoint.get("layer")
            cache_path /= "train.pt" if layer is None else f"train-L{layer}.pt"
            if not cache_path.is_file():
                raise common.ClassifierError(f"supplemental embedding cache is missing: {cache_path}")
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if not isinstance(cache, dict) or "X" not in cache:
                raise common.ClassifierError(f"supplemental cache has no X tensor: {cache_path}")
            features = cache["X"].float()
            if features.ndim != 2 or features.shape != (len(frame), checkpoint["in_dim"]):
                raise common.ClassifierError(
                    f"{species}/{locus_set}: supplemental cache dimensions/rows do not match "
                    f"parquet and checkpoint: {cache_path}"
                )
            chunks.append(features)
            paths.append(str(cache_path))
    validate_identifiers(all_names, pgb_names, species)
    if model_name == "deepcre":
        inputs: torch.Tensor | list[str] = [sequence for chunk in chunks for sequence in chunk]
    else:
        inputs = torch.cat(chunks, dim=0)  # type: ignore[arg-type]
    return inputs, {"counts": counts, "paths": paths, "total": len(all_names)}


def stratified_replay_indices(y: torch.Tensor, requested: int, generator: torch.Generator) -> torch.Tensor:
    """Sample measured PGB train genes; give high/medium-bearing genes coverage."""

    if requested <= 0 or len(y) == 0:
        raise common.ClassifierError("PGB replay must contain at least one measured training gene")
    groups = [[], [], []]
    for index, row in enumerate(y):
        valid = row[row >= 0]
        if len(valid) == 0:
            continue
        # Highest observed class prioritizes genes with a medium/high tissue.
        groups[int(valid.max())].append(index)
    requested = min(requested, sum(len(group) for group in groups))
    if requested == 0:
        raise common.ClassifierError("PGB train split has no usable tissue labels")
    shuffled = []
    for group in groups:
        if group:
            order = torch.randperm(len(group), generator=generator).tolist()
            shuffled.append([group[index] for index in order])
        else:
            shuffled.append([])
    selected: list[int] = []
    while len(selected) < requested and any(shuffled):
        for group in (2, 1, 0):
            if shuffled[group] and len(selected) < requested:
                selected.append(shuffled[group].pop())
    if len(selected) < requested:
        raise common.ClassifierError("not enough PGB training genes with usable labels for replay")
    order = torch.randperm(len(selected), generator=generator).tolist()
    return torch.tensor([selected[index] for index in order], dtype=torch.long)


def source_splits(args: argparse.Namespace, config: dict[str, object], manifest: dict[str, object], species: str):
    dataset_root = Path(str(config["dataset_root"]))
    if config["model"] == "deepcre":
        train = deepcre.load_sequence_split(dataset_root, manifest, species, "train")
        validation = deepcre.load_sequence_split(dataset_root, manifest, species, "validation")
    else:
        train = common.load_label_split(dataset_root, manifest, species, "train")
        validation = common.load_label_split(dataset_root, manifest, species, "validation")
        train_x, _, layer = common.load_embedding_split(
            Path(str(config["embed_root"])), str(config["model"]), str(config["size"]),
            species, "train", args.layer, len(train["y"]),
        )
        validation_x, _, validation_layer = common.load_embedding_split(
            Path(str(config["embed_root"])), str(config["model"]), str(config["size"]),
            species, "validation", args.layer, len(validation["y"]),
        )
        if layer != validation_layer:
            raise common.ClassifierError(f"{species}: PGB embedding layers differ across splits")
        train["X"], validation["X"] = train_x, validation_x
    return train, validation


def inputs_for_indices(
    source: torch.Tensor | list[str], indices: torch.Tensor, device: str, checkpoint: dict[str, object]
) -> torch.Tensor:
    if isinstance(source, torch.Tensor):
        x = source[indices].to(device)
        return (x - checkpoint["feature_mean"].to(device)) / checkpoint["feature_std"].to(device)
    sequences = [source[index] for index in indices.tolist()]
    return deepcre.one_hot_sequences(sequences, device)


def infer_validation(
    model: torch.nn.Module,
    validation: dict[str, object],
    checkpoint: dict[str, object],
    model_name: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    if model_name == "deepcre":
        return deepcre.infer_logits(model, validation, device, batch_size)
    return common.infer_logits(
        model, validation["X"], checkpoint["feature_mean"], checkpoint["feature_std"],
        device, batch_size,
    )


def make_model(checkpoint: dict[str, object], model_name: str, device: str) -> torch.nn.Module:
    if model_name == "deepcre":
        model = deepcre.DeepCREClassifier(
            checkpoint["sequence_length"], checkpoint["n_tissues"], checkpoint["dropout"]
        )
    else:
        model = common.TissueClassifierHead(
            checkpoint["in_dim"], checkpoint["n_tissues"],
            checkpoint["hidden_dim"], checkpoint["dropout"],
        )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def score_validation(
    model: torch.nn.Module,
    validation: dict[str, object],
    checkpoint: dict[str, object],
    model_name: str,
    device: str,
    batch_size: int,
    targets: Sequence[dict[str, object]],
    weights: torch.Tensor | None,
) -> tuple[float, float, dict[str, object]]:
    logits = infer_validation(model, validation, checkpoint, model_name, device, batch_size)
    loss = float(common.masked_cross_entropy(logits, validation["y"], weights).item())
    metrics, _ = common.score_outputs(validation["y"], logits, targets)
    macro_f1 = float(metrics["overall"]["macro_f1"])
    return macro_f1, loss, metrics


def run_species(
    args: argparse.Namespace,
    config: dict[str, object],
    manifest: dict[str, object],
    species: str,
    output_run: Path,
) -> dict[str, object]:
    destination = output_run / species
    if (destination / "finetune_metrics.json").is_file() and (destination / "best.pt").is_file():
        print(f"Skipping completed {config['model']}/{config['size']}/{species}", flush=True)
        return json.loads((destination / "finetune_metrics.json").read_text(encoding="utf-8"))

    source_checkpoint_path = args.source_run_dir / species / "best.pt"
    if not source_checkpoint_path.is_file():
        raise common.ClassifierError(f"source checkpoint is missing: {source_checkpoint_path}")
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("species") != species or checkpoint.get("model") != config["model"]:
        raise common.ClassifierError(f"source checkpoint metadata mismatch: {source_checkpoint_path}")
    if checkpoint.get("size") != config["size"]:
        raise common.ClassifierError(f"source checkpoint size mismatch: {source_checkpoint_path}")
    if checkpoint.get("num_classes") != 3 or tuple(checkpoint.get("class_names", ())) != common.CLASS_NAMES:
        raise common.ClassifierError(f"source checkpoint class contract mismatch: {source_checkpoint_path}")
    args.layer = checkpoint.get("layer")
    train, validation = source_splits(args, config, manifest, species)
    if train["y"].shape[1] != checkpoint["n_tissues"]:
        raise common.ClassifierError(f"{species}: PGB tissue count differs from source checkpoint")
    targets = manifest["species"][species]["targets"]
    supplemental, supplemental_meta = load_supplemental(
        args.supplemental_dataset_root, args.supplemental_embed_root,
        str(config["model"]), str(config["size"]), species, checkpoint, train["names"],
    )
    pgb_source = train["sequences"] if config["model"] == "deepcre" else train["X"]
    seed = common.stable_seed(args.seed, str(config["model"]), str(config["size"]), species)
    common.seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = make_model(checkpoint, str(config["model"]), args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, args.patience // 2)
    )
    weights = checkpoint.get("class_weights")
    # Keep original PGB training-derived class weights. Supplemental weak labels
    # are not used in normalization or in the class-weight calculation.
    if weights is not None and weights.shape != (checkpoint["n_tissues"], 3):
        raise common.ClassifierError(f"{species}: source checkpoint class-weight shape differs")
    weight_device = weights.to(args.device) if weights is not None else None
    pseudo_target = torch.zeros((len(supplemental), checkpoint["n_tissues"]), dtype=torch.long)
    baseline_f1, baseline_loss, _ = score_validation(
        model, validation, checkpoint, str(config["model"]), args.device,
        args.eval_batch_size, targets, weights,
    )
    best_f1 = baseline_f1
    best_epoch = 0
    best_path = destination / "best.pt"
    history: list[dict[str, object]] = []
    start_epoch = 1
    if (destination / "last.pt").is_file():
        resume = torch.load(destination / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(resume["state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        deepcre.restore_rng_state(resume["rng_state"], generator)
        start_epoch = int(resume["epoch"]) + 1
        best_f1 = float(resume["best_f1"])
        best_epoch = int(resume["best_epoch"])
        history = list(resume["history"])
        print(f"  {species}: resuming fine-tune at epoch {start_epoch}", flush=True)
    else:
        copied = dict(checkpoint)
        copied["fine_tune"] = {
            "source_checkpoint": str(source_checkpoint_path),
            "source_best_epoch": checkpoint.get("best_epoch"),
            "selected_epoch": 0,
            "weak_low_labels": True,
            "pgb_validation_macro_f1": baseline_f1,
        }
        common.save_checkpoint(copied, best_path)

    if start_epoch > args.epochs and not best_path.is_file():
        raise common.ClassifierError(f"{species}: fine-tune resume has no best checkpoint")
    usable_replay = int((train["y"] >= 0).any(dim=1).sum())
    replay_n = min(usable_replay, math.ceil(len(supplemental) * args.replay_ratio))
    if replay_n <= 0:
        raise common.ClassifierError("replay ratio gives zero measured PGB training examples")
    print(
        f"  {species}: supplemental train={len(supplemental)} "
        f"({supplemental_meta['counts']}), replay/epoch={replay_n}, "
        f"PGB validation baseline macro-F1={baseline_f1:.4f}", flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        replay_indices = stratified_replay_indices(train["y"], replay_n, generator)
        supplemental_order = torch.randperm(len(supplemental), generator=generator)
        steps = min(
            replay_n, len(supplemental),
            max(1, math.ceil((replay_n + len(supplemental)) / args.batch_size)),
        )
        replay_batches = torch.tensor_split(replay_indices, steps)
        supplemental_batches = torch.tensor_split(supplemental_order, steps)
        loss_total = 0.0
        for replay_batch, supplemental_batch in zip(replay_batches, supplemental_batches):
            optimizer.zero_grad(set_to_none=True)
            replay_x = inputs_for_indices(pgb_source, replay_batch, args.device, checkpoint)
            replay_y = train["y"][replay_batch].to(args.device)
            replay_loss = common.masked_cross_entropy(model(replay_x), replay_y, weight_device)
            supplemental_x = inputs_for_indices(
                supplemental, supplemental_batch, args.device, checkpoint
            )
            supplemental_y = pseudo_target[supplemental_batch].to(args.device)
            weak_loss = common.masked_cross_entropy(model(supplemental_x), supplemental_y)
            loss = (1 - args.supplemental_loss_weight) * replay_loss
            loss = loss + args.supplemental_loss_weight * weak_loss
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach())

        val_f1, val_loss, val_metrics = score_validation(
            model, validation, checkpoint, str(config["model"]), args.device,
            args.eval_batch_size, targets, weights,
        )
        scheduler.step(val_f1)
        improved = val_f1 > best_f1 + args.min_delta
        if improved:
            best_f1, best_epoch = val_f1, epoch
            selected = dict(checkpoint)
            selected["state_dict"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            selected["best_epoch"] = epoch
            selected["selection_metric"] = "pgb_validation_macro_f1"
            selected["selection_value"] = val_f1
            selected["fine_tune"] = {
                "source_checkpoint": str(source_checkpoint_path),
                "source_best_epoch": checkpoint.get("best_epoch"),
                "selected_epoch": epoch,
                "weak_low_labels": True,
                "pgb_validation_macro_f1": val_f1,
            }
            common.save_checkpoint(selected, best_path)
        history.append({
            "epoch": epoch,
            "train_mixed_loss": loss_total / steps,
            "pgb_validation_loss": val_loss,
            "pgb_validation_macro_f1": val_f1,
            "pgb_validation_balanced_accuracy": val_metrics["overall"]["balanced_accuracy"],
            "best_epoch": best_epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        common.save_checkpoint({
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": deepcre.capture_rng_state(generator),
            "epoch": epoch,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "history": history,
        }, destination / "last.pt")
        print(
            f"  {species}: epoch={epoch} mixed_loss={loss_total / steps:.4f} "
            f"PGB_val_macro_F1={val_f1:.4f} best={best_f1:.4f}@{best_epoch}",
            flush=True,
        )
        if epoch - best_epoch >= args.patience:
            print(f"  {species}: early stopped after {epoch} epochs", flush=True)
            break

    if history:
        pd.DataFrame(history).to_csv(destination / "finetune_history.csv", index=False)
    result = {
        "species": species,
        "model": config["model"],
        "size": config["size"],
        "source_checkpoint": str(source_checkpoint_path),
        "baseline_pgb_validation_macro_f1": baseline_f1,
        "best_pgb_validation_macro_f1": best_f1,
        "best_finetune_epoch": best_epoch,
        "supplemental_train": supplemental_meta,
        "pgb_replay_per_epoch": replay_n,
        "supplemental_loss_weight": args.supplemental_loss_weight,
        "weak_label_caveat": "supplemental loci lack measured tissue TPM; low is a weak training target",
    }
    common.write_json_atomic(result, destination / "finetune_metrics.json")
    return result


def run(args: argparse.Namespace) -> int:
    args.source_run_dir = args.source_run_dir.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.supplemental_dataset_root = args.supplemental_dataset_root.expanduser().resolve()
    args.supplemental_embed_root = args.supplemental_embed_root.expanduser().resolve()
    if args.run_dir == args.source_run_dir or args.source_run_dir in args.run_dir.parents:
        raise common.ClassifierError("fine-tune output must not be inside the source run directory")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise common.ClassifierError("CUDA requested but unavailable")
    config_path = args.source_run_dir / "run_config.json"
    if not config_path.is_file() or not (args.source_run_dir / "summary.json").is_file():
        raise common.ClassifierError(f"source classifier run is incomplete: {args.source_run_dir}")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if source_config["model"] not in {"deepcre", "agront", "evo2", "plantcad2", "ntv3"}:
        raise common.ClassifierError(f"unsupported classifier model: {source_config['model']}")
    manifest = common.load_manifest(Path(str(source_config["dataset_root"])))
    if not (args.supplemental_dataset_root / "pseudogenes").is_dir() or not (
        args.supplemental_dataset_root / "intergenic"
    ).is_dir():
        raise common.ClassifierError(
            "expected pseudogenes/ and intergenic/ under supplemental dataset root"
        )

    fine_config = dict(source_config)
    fine_config.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "supplemental_finetune",
        "source_run_dir": str(args.source_run_dir),
        "supplemental_dataset_root": str(args.supplemental_dataset_root),
        "supplemental_embed_root": str(args.supplemental_embed_root),
        "replay_ratio": args.replay_ratio,
        "supplemental_loss_weight": args.supplemental_loss_weight,
        "fine_tune_epochs": args.epochs,
        "fine_tune_learning_rate": args.learning_rate,
        "fine_tune_weight_decay": args.weight_decay,
        "fine_tune_patience": args.patience,
        "fine_tune_batch_size": args.batch_size,
        "fine_tune_eval_batch_size": args.eval_batch_size,
        "fine_tune_min_delta": args.min_delta,
        "fine_tune_seed": args.seed,
        "selection": "measured PGB validation macro-F1 only; source checkpoint is epoch 0",
        "fine_tune_data": "supplemental train only + measured PGB train replay",
        "test_use": "PGB test and supplemental test are held out until evaluation",
        "weak_label_warning": "supplemental low is presumed, not measured tissue TPM",
    })
    output_config = args.run_dir / "run_config.json"
    if output_config.exists():
        existing = json.loads(output_config.read_text(encoding="utf-8"))
        keys = (
            "source_run_dir", "supplemental_dataset_root", "supplemental_embed_root",
            "replay_ratio", "supplemental_loss_weight", "fine_tune_epochs",
            "fine_tune_learning_rate", "fine_tune_weight_decay", "fine_tune_patience",
            "fine_tune_batch_size", "fine_tune_eval_batch_size", "fine_tune_min_delta",
            "fine_tune_seed",
        )
        if any(existing.get(key) != fine_config.get(key) for key in keys):
            raise common.ClassifierError("fine-tune run already exists with different settings")
    elif args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise common.ClassifierError(f"fine-tune output is nonempty without run config: {args.run_dir}")
    else:
        common.write_json_atomic(fine_config, output_config)
    results = {}
    for species in source_config["species"]:
        print(f"Fine-tuning {source_config['model']}/{source_config['size']}/{species}", flush=True)
        results[species] = run_species(args, source_config, manifest, species, args.run_dir)
    common.write_json_atomic({
        "model": source_config["model"],
        "size": source_config["size"],
        "species": results,
        "benchmark_note": "test metrics are written by the separate evaluate commands",
    }, args.run_dir / "finetune_summary.json")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--supplemental-dataset-root", type=Path, required=True)
    parser.add_argument("--supplemental-embed-root", type=Path, required=True)
    parser.add_argument("--replay-ratio", type=float, default=1.0,
                        help="measured PGB train genes per supplemental train locus (default: 1)")
    parser.add_argument("--supplemental-loss-weight", type=float, default=0.2,
                        help="weight of weak supplemental loss; PGB replay gets 1-weight")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.replay_ratio <= 0 or not 0 < args.supplemental_loss_weight < 1:
        parser.error("--replay-ratio must be positive and --supplemental-loss-weight must be in (0,1)")
    if min(args.epochs, args.patience, args.batch_size, args.eval_batch_size) <= 0:
        parser.error("epochs, patience, and batch sizes must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.min_delta < 0:
        parser.error("learning rate must be positive; weight decay and min delta cannot be negative")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (common.ClassifierError, OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
