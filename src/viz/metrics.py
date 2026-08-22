"""Convergence curve plotting for train/val/test metric partitions (W6)."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_convergence_curves(
    metric_tracker,
    output_path: str = "artifacts/convergence.png",
) -> Figure:
    """Plot loss and accuracy convergence curves across partitions.

    Args:
        metric_tracker: Object with a `.records` dict containing the keys
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
            mapped to equal-length float lists.
        output_path: Destination PNG path; parent dirs are created.

    Returns:
        The rendered matplotlib Figure. The caller decides whether to show().
    """
    records = metric_tracker.records
    n = len(records["train_loss"])
    epochs = range(1, n + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))

    ax_loss.plot(
        epochs,
        records["train_loss"],
        label="Train Loss",
        color="#1f77b4",
        lw=2,
    )
    ax_loss.plot(
        epochs,
        records["val_loss"],
        label="Val Loss",
        color="#ff7f0e",
        lw=2,
        linestyle="--",
    )
    ax_loss.plot(
        epochs,
        records["test_loss"],
        label="Test Loss (Holdout)",
        color="#d62728",
        lw=1.5,
        linestyle=":",
    )
    ax_loss.set_title("Cross-Entropy Loss Across Partitions", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Binary Cross-Entropy")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_acc.plot(
        epochs,
        records["train_acc"],
        label="Train Accuracy",
        color="#2ca02c",
        lw=2,
    )
    ax_acc.plot(
        epochs,
        records["val_acc"],
        label="Val Accuracy",
        color="#9467bd",
        lw=2,
        linestyle="--",
    )
    ax_acc.plot(
        epochs,
        records["test_acc"],
        label="Test Accuracy (Holdout)",
        color="#d62728",
        lw=1.5,
        linestyle=":",
    )
    ax_acc.set_title("Classification Accuracy Across Partitions", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)

    return fig
