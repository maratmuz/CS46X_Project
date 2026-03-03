#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import roc_auc_score
from torch import nn
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

FEATURE_SHORT_NAMES = {
    0: "ig",
    1: "utr",
    2: "cds",
    3: "intron",
}


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


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

    # Kept for CLI compatibility with training script.
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

    p.add_argument("--checkpoint_name", default="best_checkpoint.pt")
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


def safe_ovr_auroc(y_true: np.ndarray, y_prob: np.ndarray, class_idx: int) -> float:
    binary = (y_true == class_idx).astype(np.int32)
    if np.unique(binary).size < 2:
        return float("nan")
    return float(roc_auc_score(binary, y_prob[:, class_idx]))


def per_class_aurocs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_map: Dict[int, str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    vals: List[float] = []
    for class_idx in sorted(label_map):
        name = label_map[class_idx]
        auc = safe_ovr_auroc(y_true, y_prob, class_idx)
        out[name] = auc
        if not math.isnan(auc):
            vals.append(auc)
    out["macro"] = float(np.mean(vals)) if vals else float("nan")
    return out


def sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def precision_recall_f1(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def format_ascii_table(table: List[List[str]], title: str) -> str:
    widths = [max(len(str(row[col])) for row in table) for col in range(len(table[0]))]

    def sep_line(title_text: Optional[str] = None) -> str:
        parts = []
        for i, w in enumerate(widths):
            if i == 0 and title_text is not None:
                fill = max(w + 2 - len(title_text), 0)
                parts.append(title_text + ("-" * fill))
            else:
                parts.append("-" * (w + 2))
        return "+" + "+".join(parts) + "+"

    def data_line(row: List[str]) -> str:
        cells = []
        for i, cell in enumerate(row):
            cells.append(" " + str(cell).ljust(widths[i]) + " ")
        return "|" + "|".join(cells) + "|"

    lines: List[str] = []
    lines.append(sep_line(title))
    lines.append(data_line(table[0]))
    lines.append(sep_line())
    for row in table[1:]:
        lines.append(data_line(row))
    lines.append(sep_line())
    return "\n".join(lines)


def compute_feature_confusion_artifacts(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, object]:
    cm = np.zeros((4, 4), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(t) < 4 and 0 <= int(p) < 4:
            cm[int(t), int(p)] += 1

    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_cm = np.divide(cm.astype(np.float64), row_sums, where=(row_sums != 0))
    norm_cm[row_sums.ravel() == 0] = 0.0

    scores: Dict[str, Dict[str, float]] = {}
    for class_idx in range(4):
        name = FEATURE_SHORT_NAMES[class_idx]
        not_col = np.arange(4) != class_idx
        tp = float(cm[class_idx, class_idx])
        fp = float(np.sum(cm[not_col, class_idx]))
        fn = float(np.sum(cm[class_idx, not_col]))
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        scores[name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Composite metrics adapted from Helixer ConfusionMatrixGenic.
    legacy_tp = float(cm[2, 2] + cm[3, 3] + cm[2, 3] + cm[3, 2])
    legacy_fp = float(cm[0, 2] + cm[0, 3] + cm[1, 2] + cm[1, 3])
    legacy_fn = float(cm[2, 0] + cm[3, 0] + cm[2, 1] + cm[3, 1])
    p, r, f1 = precision_recall_f1(legacy_tp, legacy_fp, legacy_fn)
    scores["legacy_cds"] = {
        "TP": legacy_tp,
        "FP": legacy_fp,
        "FN": legacy_fn,
        "precision": p,
        "recall": r,
        "f1": f1,
    }

    sub_tp = scores["cds"]["TP"] + scores["intron"]["TP"]
    sub_fp = scores["cds"]["FP"] + scores["intron"]["FP"]
    sub_fn = scores["cds"]["FN"] + scores["intron"]["FN"]
    p, r, f1 = precision_recall_f1(sub_tp, sub_fp, sub_fn)
    scores["sub_genic"] = {
        "TP": sub_tp,
        "FP": sub_fp,
        "FN": sub_fn,
        "precision": p,
        "recall": r,
        "f1": f1,
    }

    genic_tp = scores["utr"]["TP"] + scores["cds"]["TP"] + scores["intron"]["TP"]
    genic_fp = scores["utr"]["FP"] + scores["cds"]["FP"] + scores["intron"]["FP"]
    genic_fn = scores["utr"]["FN"] + scores["cds"]["FN"] + scores["intron"]["FN"]
    p, r, f1 = precision_recall_f1(genic_tp, genic_fp, genic_fn)
    scores["genic"] = {
        "TP": genic_tp,
        "FP": genic_fp,
        "FN": genic_fn,
        "precision": p,
        "recall": r,
        "f1": f1,
    }

    total_acc = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0

    cm_table: List[List[str]] = [["", "ig_pred", "utr_pred", "cds_pred", "intron_pred"]]
    for i in range(4):
        cm_table.append(
            [
                f"{FEATURE_SHORT_NAMES[i]}_ref",
                str(int(cm[i, 0])),
                str(int(cm[i, 1])),
                str(int(cm[i, 2])),
                str(int(cm[i, 3])),
            ]
        )

    norm_table: List[List[str]] = [["", "ig_pred", "utr_pred", "cds_pred", "intron_pred"]]
    for i in range(4):
        norm_table.append(
            [
                f"{FEATURE_SHORT_NAMES[i]}_ref",
                str(round(float(norm_cm[i, 0]), 4)),
                str(round(float(norm_cm[i, 1]), 4)),
                str(round(float(norm_cm[i, 2]), 4)),
                str(round(float(norm_cm[i, 3]), 4)),
            ]
        )

    f1_table: List[List[str]] = [["", "norm. H", "Precision", "Recall", "F1-Score"]]
    for key in ["ig", "utr", "cds", "intron"]:
        f1_table.append(
            [
                key,
                "",
                f"{scores[key]['precision']:.4f}",
                f"{scores[key]['recall']:.4f}",
                f"{scores[key]['f1']:.4f}",
            ]
        )
    f1_table.append(["", "", "", "", ""])
    for key in ["legacy_cds", "sub_genic", "genic"]:
        f1_table.append(
            [
                key,
                "",
                f"{scores[key]['precision']:.4f}",
                f"{scores[key]['recall']:.4f}",
                f"{scores[key]['f1']:.4f}",
            ]
        )

    return {
        "cm": cm,
        "norm_cm": norm_cm,
        "scores": scores,
        "total_acc": total_acc,
        "cm_table": cm_table,
        "norm_table": norm_table,
        "f1_table": f1_table,
    }


def main() -> None:
    metrics_start = time.time()
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    ds_obj = load_from_disk(args.dataset_dir)
    if not isinstance(ds_obj, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict at {args.dataset_dir} (with train/val/test), got {type(ds_obj)}. "
            "Run make_splits.py first."
        )
    if "test" not in ds_obj:
        raise ValueError("DatasetDict is missing required split: test")

    test_split = ds_obj["test"]
    if len(test_split) == 0:
        raise ValueError("test split is empty")

    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    if args.use_cuda and device.type != "cuda":
        print("[WARN] --use_cuda=True but CUDA is unavailable; using CPU.")

    checkpoint_path = os.path.join(args.out_dir, args.checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    embedding_dim = int(ckpt["embedding_dim"])
    dropout = float(ckpt.get("dropout", args.dropout))

    model = DualHeadMLP(embedding_dim=embedding_dim, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    loader = DataLoader(
        EmbeddingDataset(test_split),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
    )

    all_feature_logits: List[np.ndarray] = []
    all_phase_logits: List[np.ndarray] = []
    all_feature_targets: List[np.ndarray] = []
    all_phase_targets: List[np.ndarray] = []

    with torch.no_grad():
        for embs, feature_targets, phase_targets in loader:
            embs = embs.to(device)
            logits_feature, logits_phase = model(embs)

            all_feature_logits.append(logits_feature.cpu().numpy())
            all_phase_logits.append(logits_phase.cpu().numpy())
            all_feature_targets.append(feature_targets.numpy())
            all_phase_targets.append(phase_targets.numpy())

    feature_logits = np.concatenate(all_feature_logits, axis=0)
    phase_logits = np.concatenate(all_phase_logits, axis=0)
    feature_targets = np.concatenate(all_feature_targets, axis=0)
    phase_targets = np.concatenate(all_phase_targets, axis=0)

    feature_probs = torch.softmax(torch.from_numpy(feature_logits), dim=1).numpy()
    phase_probs = torch.softmax(torch.from_numpy(phase_logits), dim=1).numpy()

    feature_preds = np.argmax(feature_probs, axis=1)
    phase_preds = np.argmax(phase_probs, axis=1)

    feature_accuracy = float((feature_preds == feature_targets).mean())
    phase_accuracy = float((phase_preds == phase_targets).mean())

    feature_aurocs = per_class_aurocs(
        y_true=feature_targets,
        y_prob=feature_probs,
        label_map=FEATURE_ID_TO_NAME,
    )
    phase_aurocs = per_class_aurocs(
        y_true=phase_targets,
        y_prob=phase_probs,
        label_map=PHASE_ID_TO_NAME,
    )

    feature_conf = compute_feature_confusion_artifacts(
        y_true=feature_targets,
        y_pred=feature_preds,
    )

    confusion_report = "\n\n".join(
        [
            format_ascii_table(feature_conf["cm_table"], "confusion_matrix"),
            format_ascii_table(feature_conf["norm_table"], "normalized_confusion_matrix"),
            format_ascii_table(feature_conf["f1_table"], "F1_summary"),
        ]
    )
    confusion_report += f"\nTotal acc: {feature_conf['total_acc']:.4f}"
    confusion_report += (
        f"\n\nmetrics calculation took: {(time.time() - metrics_start) / 60.0:.2f} minutes\n"
    )

    confusion_report_path = os.path.join(args.out_dir, "feature_confusion_report.txt")
    with open(confusion_report_path, "w") as f:
        f.write(confusion_report)

    metrics = {
        "dataset_dir": args.dataset_dir,
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "test_size": int(len(test_split)),
        "feature_accuracy": feature_accuracy,
        "phase_accuracy": phase_accuracy,
        "feature_auroc_ovr": feature_aurocs,
        "phase_auroc_ovr": phase_aurocs,
        "feature_total_accuracy_from_confusion": feature_conf["total_acc"],
        "feature_confusion_matrix": feature_conf["cm"].tolist(),
        "feature_normalized_confusion_matrix": feature_conf["norm_cm"].tolist(),
        "feature_f1_summary": {
            k: {
                "precision": v["precision"],
                "recall": v["recall"],
                "f1": v["f1"],
            }
            for k, v in feature_conf["scores"].items()
        },
        "feature_confusion_report_path": confusion_report_path,
        "feature_map": FEATURE_ID_TO_NAME,
        "phase_map": PHASE_ID_TO_NAME,
    }

    out_json_path = os.path.join(args.out_dir, "test_metrics.json")
    with open(out_json_path, "w") as f:
        json.dump(sanitize_for_json(metrics), f, indent=2)

    print(f"[INFO] feature_accuracy={feature_accuracy:.6f}")
    print(f"[INFO] phase_accuracy={phase_accuracy:.6f}")
    print(f"[INFO] feature_auroc_macro={feature_aurocs['macro']}")
    print(f"[INFO] phase_auroc_macro={phase_aurocs['macro']}")
    print(confusion_report)
    print(f"[INFO] Wrote metrics: {out_json_path}")
    print(f"[INFO] Wrote confusion report: {confusion_report_path}")

    if args.wandb_project:
        if wandb is None:
            print("[WARN] wandb is not installed; skipping wandb logging.")
        else:
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "dataset_dir": args.dataset_dir,
                    "out_dir": args.out_dir,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "use_cuda": bool(args.use_cuda),
                },
            )
            run.log(
                {
                    "test/feature_accuracy": feature_accuracy,
                    "test/feature_total_accuracy_from_confusion": feature_conf["total_acc"],
                    "test/phase_accuracy": phase_accuracy,
                    "test/feature_auroc_macro": feature_aurocs["macro"],
                    "test/phase_auroc_macro": phase_aurocs["macro"],
                }
            )
            run.finish()


if __name__ == "__main__":
    main()
