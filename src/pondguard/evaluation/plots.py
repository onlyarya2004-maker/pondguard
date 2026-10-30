"""Figures for the report, drawn from the saved result files (never from typed numbers).

results/figures/confusion_<variant>.png   predicted vs labelled risk level
results/figures/metrics_ab.png            the main rule-based metrics, A next to B, with targets
results/figures/cost_quality.png          mean tokens per run vs risk accuracy, per group
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: write files only
import matplotlib.pyplot as plt  # noqa: E402

from pondguard.evaluation import metrics  # noqa: E402
from pondguard.evaluation.experiment import latest  # noqa: E402
from pondguard.evaluation.runner import eval_config, read_rows  # noqa: E402
from pondguard.paths import RESULTS_DIR  # noqa: E402

COLOURS = {"A": "#c26a3d", "B": "#2f6f9f"}
MAIN_METRICS = [
    ("risk_accuracy", "Risk accuracy"),
    ("critical_recall", "Critical recall"),
    ("top3_hit", "Top-3 hit"),
    ("action_within_one", "Action within one level"),
    ("abstention_accuracy", "Abstention accuracy"),
]


def confusion_figure(summary: dict, variant: str, out: Path) -> None:
    cm = summary["metrics"]["all"]["confusion_matrix"]
    labels = metrics.RISKS
    cols = [*labels, "none"]
    grid = [[cm[label][c] for c in cols] for label in labels]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted risk level")
    ax.set_ylabel("labelled risk level")
    ax.set_title(f"Variant {variant}: risk level, all 40 scenarios")
    top = max(max(row) for row in grid) or 1
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            ax.text(j, i, value, ha="center", va="center",
                    color="white" if value > top / 2 else "black")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def metrics_figure(sa: dict, sb: dict, out: Path) -> None:
    targets = eval_config()["targets"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.36
    for k, (variant, summary) in enumerate((("A", sa), ("B", sb))):
        values = [summary["metrics"]["all"][m] or 0 for m, _ in MAIN_METRICS]
        xs = [i + (k - 0.5) * width for i in range(len(MAIN_METRICS))]
        bars = ax.bar(xs, values, width, label=f"Variant {variant}", color=COLOURS[variant])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], fontsize=8, padding=2)
    for i, (m, _) in enumerate(MAIN_METRICS):
        if m in targets:
            ax.hlines(targets[m], i - 0.45, i + 0.45, colors="black", linestyles="dashed", lw=1)
    ax.set_xticks(range(len(MAIN_METRICS)), [label for _, label in MAIN_METRICS], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("Single-pass RAG (A) vs agentic RAG (B); dashed line = Review 2 target")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def cost_quality_figure(rows_a: list[dict], rows_b: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for variant, rows in (("A", rows_a), ("B", rows_b)):
        for group, ms in metrics.by_group(rows).items():
            group_rows = [r for r in rows if r["group"] == group]
            tokens = metrics._mean(group_rows, "tokens") or 0
            ax.scatter(tokens, ms["risk_accuracy"] or 0, color=COLOURS[variant], s=36, alpha=0.8)
            ax.annotate(group, (tokens, ms["risk_accuracy"] or 0), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")  # fmt: skip
        ax.scatter([], [], color=COLOURS[variant], label=f"Variant {variant}")
    ax.set_xlabel("mean tokens per run")
    ax.set_ylabel("risk accuracy (per scenario group)")
    ax.set_title("Cost vs quality, one point per scenario group")
    ax.set_ylim(-0.05, 1.08)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def make_figures(results_dir: Path = RESULTS_DIR) -> list[Path]:
    a_csv, a_sum = latest("A", results_dir)
    b_csv, b_sum = latest("B", results_dir)
    sa = json.loads(a_sum.read_text(encoding="utf-8"))
    sb = json.loads(b_sum.read_text(encoding="utf-8"))
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "confusion_A.png", out_dir / "confusion_B.png",
             out_dir / "metrics_ab.png", out_dir / "cost_quality.png"]  # fmt: skip
    confusion_figure(sa, "A", paths[0])
    confusion_figure(sb, "B", paths[1])
    metrics_figure(sa, sb, paths[2])
    cost_quality_figure(read_rows(a_csv), read_rows(b_csv), paths[3])
    return paths
