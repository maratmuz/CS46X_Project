#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except Exception:  # pragma: no cover - optional runtime dependency
    wandb = None


FEATURE_ID_TO_NAME = {
    0: "intergenic",
    1: "utr",
    2: "cds",
    3: "intron",
}

PHASE_ID_TO_NAME = {
    0: "0",
    1: "1",
    2: "2",
    3: "None",
}

FEATURE_LOSS_WEIGHT = 0.8
PHASE_LOSS_WEIGHT = 0.2


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def parse_class_weights_feature(raw: str) -> List[float]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("class_weights_feature cannot be empty")

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    values = [x.strip() for x in text.split(",") if x.strip()]
    if len(values) != 4:
        raise ValueError(
            f"class_weights_feature must have 4 values for [intergenic,utr,cds,intron], got {values}"
        )
    return [float(x) for x in values]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=2e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--class_weights_feature", default="0.7,1.6,1.2,1.2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_cuda", type=str2bool, default=True)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)

    return p.parse_args()


class EmbeddingDataset(Dataset):
    def __init__(self, hf_split):
        self.hf_split = hf_split

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        row = self.hf_split[idx]
        embedding = torch.tensor(row["embedding"], dtype=torch.float32)
        feature_id = int(row["feature_id"])
        phase_id = int(row["phase_id"])
        return embedding, feature_id, phase_id


def collate_batch(
    batch: Sequence[Tuple[torch.Tensor, int, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embs = torch.stack([x[0] for x in batch], dim=0)
    feature_ids = torch.tensor([x[1] for x in batch], dtype=torch.long)
    phase_ids = torch.tensor([x[2] for x in batch], dtype=torch.long)
    return embs, feature_ids, phase_ids


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualHeadMLP(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float):
        super().__init__()
        self.feature_head = MLPHead(embedding_dim, 4, dropout)
        self.phase_head = MLPHead(embedding_dim, 4, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.feature_head(x), self.phase_head(x)


def compute_losses(
    model: DualHeadMLP,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    feature_criterion: nn.Module,
    phase_criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y_feature, y_phase = batch
    x = x.to(device)
    y_feature = y_feature.to(device)
    y_phase = y_phase.to(device)

    logits_feature, logits_phase = model(x)
    feature_loss = feature_criterion(logits_feature, y_feature)
    phase_loss = phase_criterion(logits_phase, y_phase)
    total_loss = FEATURE_LOSS_WEIGHT * feature_loss + PHASE_LOSS_WEIGHT * phase_loss
    return total_loss, feature_loss, phase_loss


def run_epoch(
    model: DualHeadMLP,
    loader: DataLoader,
    device: torch.device,
    feature_criterion: nn.Module,
    phase_criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    n_examples = 0
    total_loss_sum = 0.0
    feature_loss_sum = 0.0
    phase_loss_sum = 0.0

    for batch in loader:
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            total_loss, feature_loss, phase_loss = compute_losses(
                model=model,
                batch=batch,
                device=device,
                feature_criterion=feature_criterion,
                phase_criterion=phase_criterion,
            )

            if train_mode:
                total_loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        bs = batch[0].shape[0]
        n_examples += bs
        total_loss_sum += float(total_loss.detach().item()) * bs
        feature_loss_sum += float(feature_loss.detach().item()) * bs
        phase_loss_sum += float(phase_loss.detach().item()) * bs

    if n_examples == 0:
        return {"total_loss": math.nan, "feature_loss": math.nan, "phase_loss": math.nan}

    return {
        "total_loss": total_loss_sum / n_examples,
        "feature_loss": feature_loss_sum / n_examples,
        "phase_loss": phase_loss_sum / n_examples,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed)

    feature_class_weights = parse_class_weights_feature(args.class_weights_feature)

    ds_obj = load_from_disk(args.dataset_dir)
    if not isinstance(ds_obj, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict at {args.dataset_dir} (with train/val/test), got {type(ds_obj)}. "
            "Run make_splits.py first."
        )
    for split in ("train", "val", "test"):
        if split not in ds_obj:
            raise ValueError(f"DatasetDict is missing required split: {split}")

    train_split = ds_obj["train"]
    val_split = ds_obj["val"]

    if len(train_split) == 0:
        raise ValueError("train split is empty")
    if len(val_split) == 0:
        raise ValueError("val split is empty")

    embedding_dim = len(train_split[0]["embedding"])

    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    if args.use_cuda and device.type != "cuda":
        print("[WARN] --use_cuda=True but CUDA is unavailable; using CPU.")

    train_loader = DataLoader(
        EmbeddingDataset(train_split),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        EmbeddingDataset(val_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )

    model = DualHeadMLP(embedding_dim=embedding_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    feature_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(feature_class_weights, dtype=torch.float32, device=device)
    )
    phase_criterion = nn.CrossEntropyLoss()

    run_config = {
        "dataset_dir": args.dataset_dir,
        "out_dir": args.out_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "class_weights_feature": feature_class_weights,
        "feature_loss_weight": FEATURE_LOSS_WEIGHT,
        "phase_loss_weight": PHASE_LOSS_WEIGHT,
        "seed": args.seed,
        "use_cuda": bool(args.use_cuda),
        "device": str(device),
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "embedding_dim": embedding_dim,
        "feature_map": FEATURE_ID_TO_NAME,
        "phase_map": PHASE_ID_TO_NAME,
    }

    wandb_run = None
    if args.wandb_project:
        if wandb is None:
            print("[WARN] wandb is not installed; proceeding without wandb logging.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=run_config,
            )

    best_val_total = float("inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []

    best_ckpt_path = os.path.join(args.out_dir, "best_checkpoint.pt")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            feature_criterion=feature_criterion,
            phase_criterion=phase_criterion,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                feature_criterion=feature_criterion,
                phase_criterion=phase_criterion,
                optimizer=None,
            )

        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_feature_loss": train_metrics["feature_loss"],
            "train_phase_loss": train_metrics["phase_loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_feature_loss": val_metrics["feature_loss"],
            "val_phase_loss": val_metrics["phase_loss"],
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_total={row['train_total_loss']:.6f} val_total={row['val_total_loss']:.6f} "
            f"train_feat={row['train_feature_loss']:.6f} val_feat={row['val_feature_loss']:.6f} "
            f"train_phase={row['train_phase_loss']:.6f} val_phase={row['val_phase_loss']:.6f}"
        )

        if wandb_run is not None:
            wandb.log(row, step=epoch)

        if row["val_total_loss"] < best_val_total:
            best_val_total = row["val_total_loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": embedding_dim,
                    "dropout": args.dropout,
                    "feature_num_classes": 4,
                    "phase_num_classes": 4,
                    "best_epoch": best_epoch,
                    "best_val_total_loss": best_val_total,
                    "config": run_config,
                },
                best_ckpt_path,
            )

    summary = {
        "config": run_config,
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val_total,
        "history": history,
        "best_checkpoint_path": best_ckpt_path,
    }

    summary_path = os.path.join(args.out_dir, "train_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_val_total_loss"] = best_val_total
        wandb.summary["best_checkpoint_path"] = best_ckpt_path
        wandb.finish()

    print(f"[INFO] Best checkpoint: {best_ckpt_path}")
    print(f"[INFO] Training summary: {summary_path}")


if __name__ == "__main__":
    main()

