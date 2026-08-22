"""Render the compact ColoPola result figure used by the project README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results" / "selected_results.json"
OUTPUT = ROOT / "docs" / "assets" / "colopola_flow_augmentation.png"

GREEN = "#009E73"
GREY = "#6B7280"
GRID = "#D1D5DB"


def main() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign = payload["colopola_flow_augmentation"]
    rows = campaign["regimes"]

    x = np.arange(len(rows))
    labels = [f"{100 * row['label_fraction']:.0f}" for row in rows]
    baseline = np.asarray([row["without_augmentation_mean"] for row in rows])
    baseline_sd = np.asarray([row["without_augmentation_sd"] for row in rows])
    augmented = np.asarray([row["flow_matching_mean"] for row in rows])
    augmented_sd = np.asarray([row["flow_matching_sd"] for row in rows])
    gain = 100.0 * np.asarray([row["paired_gain"] for row in rows])
    gain_sd = 100.0 * np.asarray([row["paired_gain_sd"] for row in rows])

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True)

    axes[0].errorbar(
        x,
        baseline,
        yerr=baseline_sd,
        marker="o",
        color=GREY,
        capsize=3,
        linewidth=1.5,
        label="No augmentation",
    )
    axes[0].errorbar(
        x,
        augmented,
        yerr=augmented_sd,
        marker="s",
        color=GREEN,
        capsize=3,
        linewidth=1.7,
        label="Flow Matching augmentation",
    )
    axes[0].set_ylim(0.68, 0.98)
    axes[0].set_ylabel("Test AUC")
    axes[0].set_title("Absolute performance")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].errorbar(
        x,
        gain,
        yerr=gain_sd,
        marker="s",
        color=GREEN,
        ecolor=GREEN,
        capsize=3,
        linewidth=1.7,
    )
    axes[1].axhline(0.0, color=GREY, linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Paired AUC gain (points)")
    axes[1].set_title("Flow Matching minus no augmentation")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Labelled training roots (%)")
        axis.grid(color=GRID, linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("ColoPola: augmentation across label budgets", fontsize=12, y=0.99)
    figure.text(
        0.5,
        0.012,
        "Root-grouped split · 46 test roots / 113 acquisitions · 30 paired seeds · mean ± SD",
        ha="center",
        fontsize=7.5,
        color="#374151",
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.24, wspace=0.30)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=220, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(figure)


if __name__ == "__main__":
    main()
