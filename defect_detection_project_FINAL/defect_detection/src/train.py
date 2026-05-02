"""
train.py — Training Script
===========================
Supports: CustomCNN, ResNet-50, YOLOv8
Run:  python src/train.py --model resnet50 --epochs 50
"""

import os
import sys
import time
import argparse
import yaml
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from dataset import create_dataloaders, CLASS_NAMES
from models import get_model, load_checkpoint


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> tuple:
    """Run one full training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:>3} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    split: str = "Val",
) -> tuple:
    """Evaluate model on val/test set. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"       [{split}] ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def save_checkpoint(model, optimizer, epoch, val_acc, save_path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }, save_path)


def plot_training_curves(history: dict, save_dir: Path):
    """Plot and save loss + accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-o", markersize=4, label="Val Acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: training_curves.png")


# ── Main Training Function ────────────────────────────────────────────────────

def train_classifier(args):
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"\n{'='*55}")
    print(f"  Defect Detection — Training ({args.model.upper()})")
    print(f"{'='*55}")

    # Create output dirs
    save_dir = Path(args.save_dir)
    plots_dir = Path(args.plots_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    print("\n📂 Loading data...")
    loaders = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    print("\n🧠 Initializing model...")
    model = get_model(
        model_name=args.model,
        num_classes=args.num_classes,
        dropout=args.dropout,
        freeze_layers=True,
        device=device,
    )

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    print(f"\n🚀 Training for {args.epochs} epochs...\n")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze all layers at epoch 10 for fine-tuning (ResNet)
        if epoch == 10 and hasattr(model, "unfreeze_all"):
            model.unfreeze_all()
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch, eta_min=1e-7)
            print("  🔓 All layers unfrozen for fine-tuning")

        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, loaders["val"], criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Print epoch summary
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:>3}/{args.epochs}  "
              f"Loss: {train_loss:.4f}/{val_loss:.4f}  "
              f"Acc: {train_acc:.1f}%/{val_acc:.1f}%  "
              f"LR: {lr:.2e}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = save_dir / f"best_{args.model}.pth"
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"  ✅ New best! Val Acc: {val_acc:.2f}% — Saved to {checkpoint_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Plot curves
    plot_training_curves(history, plots_dir)

    # Final test evaluation
    print("\n📊 Evaluating on test set...")
    best_model = get_model(args.model, args.num_classes, device=device)
    best_model = load_checkpoint(best_model, str(save_dir / f"best_{args.model}.pth"), device)
    test_loss, test_acc = validate(best_model, loaders["test"], criterion, device, split="Test")
    print(f"\n  Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.2f}%")
    print(f"\n✅ Training complete! Best Val Acc: {best_val_acc:.2f}%")


# ── YOLOv8 Training ───────────────────────────────────────────────────────────

def train_yolo(args):
    """Train YOLOv8 for defect object detection."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        return

    print("\n🔵 Training YOLOv8 for Defect Detection...")

    model = YOLO(f"yolov8{args.yolo_size}.pt")
    results = model.train(
        data=args.yolo_data,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        device=0 if torch.cuda.is_available() else "cpu",
        project="results",
        name="yolov8_defects",
        exist_ok=True,
        augment=True,
        mosaic=0.5,
        lr0=0.01,
        lrf=0.001,
        patience=args.patience,
        save=True,
        plots=True,
        verbose=True,
    )
    print(f"\n✅ YOLOv8 training complete! Results saved to results/yolov8_defects/")


# ── Argument Parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Defect Detection Models")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Dataset directory")
    parser.add_argument("--save_dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--plots_dir", type=str, default="results/plots", help="Plots directory")

    # Model
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["custom_cnn", "resnet50", "efficientnet", "yolov8"])
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.4)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    # YOLO specific
    parser.add_argument("--yolo_data", type=str, default="configs/dataset.yaml")
    parser.add_argument("--yolo_size", type=str, default="n", choices=["n", "s", "m", "l", "x"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model == "yolov8":
        train_yolo(args)
    else:
        train_classifier(args)
