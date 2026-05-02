"""
evaluate.py — Model Evaluation & Metrics
=========================================
Generates: Confusion Matrix, Classification Report,
           ROC Curve, Per-Class Accuracy Bar Chart
Run: python src/evaluate.py --model resnet50 --checkpoint models/best_resnet50.pth
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from dataset import create_dataloaders, CLASS_NAMES, IDX_TO_CLASS
from models import get_model, load_checkpoint


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, device):
    """Collect all predictions and ground truth labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


# ── Plot Functions ────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Count")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    axes[0].tick_params(axis="x", rotation=30)

    # Percentage
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5)
    axes[1].set_title("Percentage (%)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: confusion_matrix.png")


def plot_per_class_accuracy(y_true, y_pred, class_names, save_dir):
    """Bar chart of accuracy per class."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    colors = ["#185FA5" if acc >= 85 else "#854F0B" if acc >= 70 else "#993556"
              for acc in per_class_acc]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, per_class_acc, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="90% target")
    ax.set_title("Per-Class Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_dir / "per_class_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: per_class_accuracy.png")


def plot_roc_curves(y_true, y_probs, class_names, save_dir):
    """ROC curves for each class (One-vs-Rest)."""
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#185FA5", "#0F6E56", "#854F0B", "#993556", "#712B13", "#27500A"]

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")
    ax.set_title("ROC Curves — One vs Rest", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: roc_curves.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--checkpoint", type=str, default="models/best_resnet50.pth")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--save_dir", type=str, default="results/plots")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*50}")
    print(f"  Defect Detection — Evaluation")
    print(f"{'='*50}")

    # Load model
    print("\n🧠 Loading model...")
    model = get_model(args.model, args.num_classes, device=device)
    model = load_checkpoint(model, args.checkpoint, device)

    # Load test data
    print("\n📂 Loading test data...")
    loaders = create_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=2)

    # Get predictions
    print("\n🔍 Running inference...")
    y_pred, y_true, y_probs = get_predictions(model, loaders["test"], device)

    # Metrics
    class_names = CLASS_NAMES[:args.num_classes]
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    overall_acc = (y_pred == y_true).mean() * 100
    print(f"  Overall Accuracy: {overall_acc:.2f}%")

    # Plots
    print("\n📈 Generating evaluation plots...")
    plot_confusion_matrix(y_true, y_pred, class_names, save_dir)
    plot_per_class_accuracy(y_true, y_pred, class_names, save_dir)
    plot_roc_curves(y_true, y_probs, class_names, save_dir)

    print(f"\n✅ Evaluation complete! Plots saved to: {save_dir}/")


if __name__ == "__main__":
    main()
