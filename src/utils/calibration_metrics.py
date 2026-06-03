"""
Calibration metrics: ECE, reliability diagrams, confidence histograms.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute_ece(y_true, y_probs, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Args:
        y_true: True labels (class indices)
        y_probs: Predicted probabilities (n_samples, n_classes)
        n_bins: Number of bins for ECE computation

    Returns:
        ECE score (float, 0-1)
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # Get predicted class and confidence
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    # Bin by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.sum() == 0:
            continue

        bin_confidences = confidences[mask]
        bin_corrects = (y_pred[mask] == y_true[mask]).astype(float)

        bin_acc = bin_corrects.mean()
        bin_conf = bin_confidences.mean()

        ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(y_true)

    return float(ece)


def compute_confidence_stats(y_true, y_probs, class_names=None):
    """
    Compute confidence statistics per class.

    Returns:
        Dict of confidence stats per class
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(y_probs.shape[1])]

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    stats = {}
    for i, cls in enumerate(class_names):
        cls_mask = y_true == i
        if cls_mask.sum() == 0:
            continue

        correct_mask = cls_mask & (y_pred == y_true)
        incorrect_mask = cls_mask & (y_pred != y_true)

        stats[cls] = {
            "mean_confidence_correct": float(confidences[correct_mask].mean()) if correct_mask.sum() > 0 else 0,
            "mean_confidence_incorrect": float(confidences[incorrect_mask].mean()) if incorrect_mask.sum() > 0 else 0,
            "std_confidence": float(confidences[cls_mask].std()),
            "n_samples": int(cls_mask.sum()),
        }

    return stats


def plot_reliability_diagram(y_true, y_probs, run_name, results_dir=None, n_bins=10):
    """
    Plot reliability diagram (confidence vs accuracy).

    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        run_name: Run name for file saving
        results_dir: Where to save the plot
        n_bins: Number of bins
    """
    if results_dir is None:
        results_dir = Path.cwd() / "outputs" / "results" / run_name

    results_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    # Compute bin statistics
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:  # Last bin includes 1.0
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])

        if mask.sum() > 0:
            bin_acc = (y_pred[mask] == y_true[mask]).mean()
            bin_conf = confidences[mask].mean()
        else:
            bin_acc = 0
            bin_conf = bin_centers[i]

        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_counts.append(mask.sum())

    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect calibration")

    # Actual calibration
    colors = ["#FF6B6B" if acc < conf else "#4ECDC4" for acc, conf in zip(bin_accs, bin_confs)]
    sizes = [max(50, c / 10) for c in bin_counts]
    ax.scatter(bin_confs, bin_accs, s=sizes, c=colors, alpha=0.7, edgecolors="black", lw=1.5)

    # Histogram of confidences (secondary axis)
    ax2 = ax.twiny()
    ax2.hist(confidences, bins=30, alpha=0.2, color="gray", label="Confidence distribution")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Confidence (histogram)", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Calibration Diagram — {run_name}", fontweight="bold", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(results_dir / "calibration_reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confidence_histogram(y_true, y_probs, run_name, class_names=None, results_dir=None):
    """
    Plot confidence distribution per class.
    """
    if results_dir is None:
        results_dir = Path.cwd() / "outputs" / "results" / run_name

    if class_names is None:
        class_names = ["NORM", "MI", "AFIB", "TACHY"]

    results_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        cls_idx = i
        cls_mask = y_true == cls_idx
        correct_mask = cls_mask & (y_pred == y_true)
        incorrect_mask = cls_mask & (y_pred != y_true)

        ax = axes[i]

        if correct_mask.sum() > 0:
            ax.hist(confidences[correct_mask], bins=30, alpha=0.7, color=color,
                   label=f"Correct (n={correct_mask.sum()})", density=True)
        if incorrect_mask.sum() > 0:
            ax.hist(confidences[incorrect_mask], bins=30, alpha=0.5, color="gray",
                   label=f"Incorrect (n={incorrect_mask.sum()})", density=True)

        ax.set_xlabel("Predicted Confidence", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{cls} Confidence Distribution", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    plt.suptitle(f"Confidence Histograms — {run_name}", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(results_dir / "calibration_confidence_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_confidence_predictions(y_true, y_probs, filepaths, labels, run_name,
                                class_names=None, results_dir=None):
    """
    Save predictions with confidence scores to CSV.
    """
    if results_dir is None:
        results_dir = Path.cwd() / "outputs" / "results" / run_name

    if class_names is None:
        class_names = ["NORM", "MI", "AFIB", "TACHY"]

    results_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)

    df = pd.DataFrame({
        "filepath": filepaths,
        "true_label": [class_names[i] for i in y_true],
        "pred_label": [class_names[i] for i in y_pred],
        "confidence": confidences,
        "correct": (y_pred == y_true).astype(int),
    })

    df.to_csv(results_dir / "predictions_with_confidence.csv", index=False)
