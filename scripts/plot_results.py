"""Render the README charts from a finished training run.

    python scripts/plot_results.py --log results/reports/training_log.jsonl \
        --eval results/reports/eval.json --out docs/img

Writes:
  training_curves.png   log-loss (train vs validation), validation accuracy, validation Brier, per epoch
  eval_comparison.png   base Qwen vs this model on the test split: accuracy, log-loss, ECE per question type

Colors: the dataviz reference palette's first two categorical slots (blue = this model / validation,
orange = base model / training), which pass the palette validator for CVD and contrast on the light surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"    # this model / validation
ORANGE = "#eb6834"  # base model / training

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def epochs_from_log(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if "epoch" in r and "val_ce" in r]


def line(ax, xs, ys, color, label, direct_label=True):
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    if direct_label and ys:
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT_2)


def training_curves(epochs: list[dict], out: Path) -> None:
    best = min(epochs, key=lambda r: r["val_ce"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    ax = axes[0]
    tr = [r for r in epochs if "train_ce" in r]
    line(ax, [r["epoch"] for r in tr], [r["train_ce"] for r in tr], ORANGE, "train")
    line(ax, [r["epoch"] for r in epochs], [r["val_ce"] for r in epochs], BLUE, "validation")
    ax.set_title("Log-loss (lower is better)")

    ax = axes[1]
    line(ax, [r["epoch"] for r in epochs], [100 * r["val_acc"] for r in epochs], BLUE, "validation",
         direct_label=False)
    ax.set_title("Validation accuracy, % (higher is better)")

    ax = axes[2]
    line(ax, [r["epoch"] for r in epochs], [r["val_brier"] for r in epochs], BLUE, "validation",
         direct_label=False)
    ax.set_title("Validation Brier (lower is better)")

    for ax in axes:
        ax.axvline(best["epoch"], color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xlabel("epoch (0 = base model before training)")
        ax.set_xticks([r["epoch"] for r in epochs])
    axes[0].annotate(f"best epoch {best['epoch']}\n(published)", (best["epoch"], axes[0].get_ylim()[1]),
                     xytext=(4, -4), textcoords="offset points", va="top", fontsize=9, color=TEXT_2)
    axes[0].margins(x=0.12)
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)


def eval_comparison(results: list[dict], out: Path) -> None:
    base = next(r for r in results if r["model"].startswith("Qwen/") or "Qwen3.5-0.8B-Base" in r["model"])
    tuned = next(r for r in results if r is not base)
    kinds = [k for k in ("choice", "noul", "score") if k in tuned["by_type"]]
    metrics = [("acc", "Accuracy, % (higher is better)", 100), ("logloss", "Log-loss (lower is better)", 1),
               ("ece", "Calibration error, ECE (lower is better)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    width = 0.38
    for ax, (key, title, scale) in zip(axes, metrics):
        xs = range(len(kinds))
        ax.bar([x - width / 2 for x in xs], [scale * base["by_type"][k][key] for k in kinds], width,
               color=ORANGE, label="base Qwen3.5-0.8B", edgecolor=SURFACE, linewidth=2)
        ax.bar([x + width / 2 for x in xs], [scale * tuned["by_type"][k][key] for k in kinds], width,
               color=BLUE, label="this model", edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(list(xs), kinds)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2)
    fig.savefig(out / "eval_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="training_log.jsonl from the run")
    ap.add_argument("--eval", required=True, help="eval.json from the run")
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    training_curves(epochs_from_log(Path(args.log)), out)
    eval_comparison(json.loads(Path(args.eval).read_text()), out)
    print(f"wrote {out / 'training_curves.png'} and {out / 'eval_comparison.png'}")


if __name__ == "__main__":
    main()
