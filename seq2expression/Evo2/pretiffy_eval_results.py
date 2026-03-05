import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

DARK   = "#0d0d0f"
PANEL  = "#16161a"
ACCENT = "#00e5ff"
MUTED  = "#4a4a5a"
WHITE  = "#e8e8f0"
GOOD   = "#00e5a0"
BAD    = "#ff4060"

CMAP = LinearSegmentedColormap.from_list("cm", [DARK, ACCENT])

plt.rcParams.update({
    "font.family":      "monospace",
    "text.color":       WHITE,
    "axes.facecolor":   PANEL,
    "figure.facecolor": DARK,
    "axes.edgecolor":   MUTED,
    "axes.labelcolor":  WHITE,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "grid.color":       MUTED,
    "grid.alpha":       0.3,
})

METRIC_KEYS = ["r2", "auroc", "accuracy", "precision", "recall", "f1"]


def load_results(run_dir: Path) -> dict:
    results = {}
    for d in run_dir.iterdir():
        if d.is_dir() and (d / "results.json").exists():
            with open(d / "results.json") as f:
                results[d.name] = json.load(f)
    return results


def plot_metrics_bar(results: dict, out_dir: Path):
    species = list(results.keys())
    x       = np.arange(len(METRIC_KEYS))
    n       = len(species)
    width   = 0.7 / n

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(DARK)

    colors = [matplotlib.colormaps["cool"](i / max(n - 1, 1)) for i in range(n)]

    for i, (sp, color) in enumerate(zip(species, colors)):
        vals = [results[sp].get(k, 0) for k in METRIC_KEYS]
        bars = ax.bar(x + i * width, vals, width, label=sp.replace("_", " "), color=color, alpha=0.85)

    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels([k.upper() for k in METRIC_KEYS], fontsize=9)
    # ax.set_ylim(0, 1.1)
    min_val = min(results[sp].get("r2", 0) for sp in species)
    ax.set_ylim(min(min_val - 0.05, 0), 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics by Species", color=WHITE, fontsize=12, pad=14)
    ax.legend(fontsize=7, framealpha=0.2, loc="upper right")
    ax.axhline(1.0, color=MUTED, linewidth=0.5, linestyle="--")
    ax.grid(axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results: dict, out_dir: Path):
    species = list(results.keys())
    n       = len(species)
    ncols   = min(n, 3)
    nrows   = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for i, sp in enumerate(species):
        cm  = np.array(results[sp]["confusion_matrix"])
        ax  = axes[i]
        im  = ax.imshow(cm, cmap=CMAP, aspect="auto", origin="lower")
        # im  = ax.imshow(cm, cmap=CMAP, aspect="auto")

        n_brackets = cm.shape[0]
        labels     = [f"B{b}" for b in range(n_brackets)]
        ax.set_xticks(range(n_brackets))
        ax.set_yticks(range(n_brackets))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual",    fontsize=8)
        ax.set_title(sp.replace("_", " "), color=WHITE, fontsize=9)

        for r in range(n_brackets):
            for c in range(n_brackets):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color=WHITE if cm[r, c] < cm.max() * 0.6 else DARK, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices (flattened across tissues)", color=WHITE, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radar(results: dict, out_dir: Path):
    species = list(results.keys())
    keys    = ["r2", "auroc", "accuracy", "precision", "recall", "f1"]
    n_keys  = len(keys)
    angles  = np.linspace(0, 2 * np.pi, n_keys, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(PANEL)
    ax.spines["polar"].set_color(MUTED)

    colors = [matplotlib.colormaps["cool"](i / max(len(species) - 1, 1)) for i in range(len(species))]

    for sp, color in zip(species, colors):
        vals  = [results[sp].get(k, 0) for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=1.5, label=sp.replace("_", " "))
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.upper() for k in keys], color=WHITE, fontsize=9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelcolor=MUTED, labelsize=6)
    ax.set_title("Metric Radar", color=WHITE, fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7, framealpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / "radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tp_fp_fn(results: dict, out_dir: Path):
    species = list(results.keys())
    fig, axes = plt.subplots(1, len(species), figsize=(5 * len(species), 4), sharey=False)
    if len(species) == 1:
        axes = [axes]

    for ax, sp in zip(axes, species):
        r  = results[sp]
        tp = np.array(r["tp"], dtype=float)
        fp = np.array(r["fp"], dtype=float)
        fn = np.array(r["fn"], dtype=float)
        x  = np.arange(len(tp))

        ax.bar(x - 0.25, tp, 0.25, label="TP", color=GOOD,   alpha=0.85)
        ax.bar(x,        fp, 0.25, label="FP", color=ACCENT,  alpha=0.85)
        ax.bar(x + 0.25, fn, 0.25, label="FN", color=BAD,     alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"B{i}" for i in range(len(tp))], fontsize=8)
        ax.set_title(sp.replace("_", " "), color=WHITE, fontsize=9)
        ax.legend(fontsize=7, framealpha=0.2)
        ax.grid(axis="y")

    fig.suptitle("TP / FP / FN per Bracket", color=WHITE, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "tp_fp_fn.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_dir = Path('shared/evo2_gene_exp/training_runs/evo2_7b_2026-03-05_11-13-32')
    results = load_results(run_dir)

    if not results:
        print("No results.json files found.")
        exit(1)

    out_dir = run_dir / "prettified_results"
    out_dir.mkdir(exist_ok=True)

    plot_metrics_bar(results, out_dir)
    plot_confusion_matrices(results, out_dir)
    plot_radar(results, out_dir)
    plot_tp_fp_fn(results, out_dir)

    print(f"saved → {out_dir}")