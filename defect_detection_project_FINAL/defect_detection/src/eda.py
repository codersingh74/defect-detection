"""
eda.py — Exploratory Data Analysis for Defect Detection Dataset
================================================================
Run: python src/eda.py --data_dir data/raw/
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = ["#185FA5", "#0F6E56", "#854F0B", "#993556", "#712B13", "#27500A"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset_info(data_dir: str) -> pd.DataFrame:
    """Recursively scan directory and collect image metadata."""
    data_dir = Path(data_dir)
    records = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))

        for img_path in tqdm(image_files, desc=f"Loading {class_name}", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            records.append({
                "path": str(img_path),
                "class": class_name,
                "filename": img_path.name,
                "height": h,
                "width": w,
                "aspect_ratio": round(w / h, 3),
                "brightness": round(float(gray.mean()), 3),
                "contrast": round(float(gray.std()), 3),
                "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
                "file_size_kb": round(img_path.stat().st_size / 1024, 2),
            })

    return pd.DataFrame(records)


def plot_class_distribution(df: pd.DataFrame, save_dir: Path):
    """Bar chart of class counts + pie chart."""
    counts = df["class"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Class Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=PALETTE[:len(counts)], edgecolor="white", linewidth=0.8)
    axes[0].set_title("Image Count per Class")
    axes[0].set_xlabel("Defect Class")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=PALETTE[:len(counts)], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    plt.savefig(save_dir / "01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 01_class_distribution.png")


def plot_image_statistics(df: pd.DataFrame, save_dir: Path):
    """Brightness, Contrast, Sharpness distributions per class."""
    metrics = ["brightness", "contrast", "sharpness"]
    titles = ["Pixel Brightness (mean)", "Contrast (std dev)", "Sharpness (Laplacian var)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Image Statistics per Class", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        for i, cls in enumerate(df["class"].unique()):
            subset = df[df["class"] == cls][metric]
            ax.hist(subset, bins=20, alpha=0.6, label=cls, color=PALETTE[i % len(PALETTE)])
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "02_image_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 02_image_statistics.png")


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: Path):
    """Correlation between numeric image features."""
    numeric_cols = ["brightness", "contrast", "sharpness", "file_size_kb", "width", "height"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", mask=mask,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 10})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 03_correlation_heatmap.png")


def plot_sample_images(df: pd.DataFrame, save_dir: Path, n_samples: int = 5):
    """Show sample images for each class."""
    classes = df["class"].unique()
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 3, n_classes * 3))
    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold", y=1.01)

    for row, cls in enumerate(classes):
        class_df = df[df["class"] == cls].sample(min(n_samples, len(df[df["class"] == cls])), random_state=42)
        for col, (_, record) in enumerate(class_df.iterrows()):
            ax = axes[row, col] if n_classes > 1 else axes[col]
            img = cv2.imread(record["path"])
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=11, fontweight="bold", rotation=0,
                               labelpad=60, va="center")
            ax.set_title(f"B:{record['brightness']:.0f} C:{record['contrast']:.0f}",
                         fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(save_dir / "04_sample_images.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 04_sample_images.png")


def plot_pixel_intensity(df: pd.DataFrame, save_dir: Path, n_samples: int = 3):
    """Pixel intensity histograms: normal vs defective comparison."""
    good_samples = df[df["class"] == "good"].head(n_samples)
    defect_samples = df[df["class"] != "good"].head(n_samples)

    fig, axes = plt.subplots(2, n_samples, figsize=(14, 7))
    fig.suptitle("Pixel Intensity: Good vs Defective", fontsize=13, fontweight="bold")

    for col, (_, record) in enumerate(good_samples.iterrows()):
        ax = axes[0, col]
        img = cv2.imread(record["path"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            ax.hist(img.flatten(), bins=64, color="#185FA5", alpha=0.8, edgecolor="none")
        ax.set_title(f"Good #{col+1}", fontsize=10)
        ax.set_xlabel("Pixel Value")
        if col == 0:
            ax.set_ylabel("Frequency", fontsize=10)

    for col, (_, record) in enumerate(defect_samples.iterrows()):
        ax = axes[1, col]
        img = cv2.imread(record["path"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            ax.hist(img.flatten(), bins=64, color="#993556", alpha=0.8, edgecolor="none")
        ax.set_title(f"Defect ({record['class']}) #{col+1}", fontsize=10)
        ax.set_xlabel("Pixel Value")
        if col == 0:
            ax.set_ylabel("Frequency", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_dir / "05_pixel_intensity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 05_pixel_intensity.png")


def plot_boxplots(df: pd.DataFrame, save_dir: Path):
    """Boxplots of brightness/contrast per class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Brightness & Contrast Distribution by Class", fontsize=13, fontweight="bold")

    sns.boxplot(data=df, x="class", y="brightness", palette=PALETTE, ax=axes[0])
    axes[0].set_title("Brightness per Class")
    axes[0].tick_params(axis="x", rotation=30)

    sns.boxplot(data=df, x="class", y="contrast", palette=PALETTE, ax=axes[1])
    axes[1].set_title("Contrast per Class")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_dir / "06_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 06_boxplots.png")


def generate_eda_report(df: pd.DataFrame) -> str:
    """Print a text summary of key findings."""
    lines = []
    lines.append("=" * 60)
    lines.append("  EXPLORATORY DATA ANALYSIS — SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append(f"\n  Total Images    : {len(df)}")
    lines.append(f"  Total Classes   : {df['class'].nunique()}")
    lines.append(f"  Avg Resolution  : {df['width'].mean():.0f} x {df['height'].mean():.0f} px")
    lines.append(f"  Avg Brightness  : {df['brightness'].mean():.1f}")
    lines.append(f"  Avg Contrast    : {df['contrast'].mean():.1f}")
    lines.append(f"  Avg Sharpness   : {df['sharpness'].mean():.1f}")
    lines.append("\n  Class Counts:")
    for cls, cnt in df["class"].value_counts().items():
        pct = cnt / len(df) * 100
        lines.append(f"    {cls:<20} {cnt:>5}  ({pct:.1f}%)")
    lines.append("\n  Class Imbalance Check:")
    max_cnt = df["class"].value_counts().max()
    min_cnt = df["class"].value_counts().min()
    ratio = max_cnt / min_cnt if min_cnt > 0 else float("inf")
    lines.append(f"    Max/Min ratio: {ratio:.2f}x")
    if ratio > 3:
        lines.append("    ⚠️  Significant imbalance — use WeightedRandomSampler!")
    else:
        lines.append("    ✅ Balanced dataset")
    lines.append("=" * 60)
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDA for Defect Detection Dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to dataset directory")
    parser.add_argument("--save_dir", type=str, default="results/plots", help="Where to save plots")
    parser.add_argument("--n_samples", type=int, default=5, help="Samples per class for visualization")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"⚠️  Data directory not found: {data_dir}")
        print("   Creating dummy data for demo...")
        _create_dummy_data(data_dir)

    print("\n📊 Loading dataset info...")
    df = load_dataset_info(str(data_dir))

    if df.empty:
        print("❌ No images found! Check your data_dir path.")
        return

    print(f"   Found {len(df)} images across {df['class'].nunique()} classes.\n")

    print("📈 Generating EDA plots...")
    plot_class_distribution(df, save_dir)
    plot_image_statistics(df, save_dir)
    plot_correlation_heatmap(df, save_dir)
    plot_sample_images(df, save_dir, n_samples=args.n_samples)
    plot_pixel_intensity(df, save_dir)
    plot_boxplots(df, save_dir)

    # Save metadata CSV
    csv_path = save_dir / "dataset_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: dataset_metadata.csv")

    print("\n" + generate_eda_report(df))
    print(f"\n✅ All plots saved to: {save_dir}/")


def _create_dummy_data(data_dir: Path):
    """Create synthetic dummy images for testing when no real dataset exists."""
    import random
    classes = ["good", "scratch", "dent", "crack", "contamination", "bent"]
    for cls in classes:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            img = np.random.randint(80, 200, (224, 224, 3), dtype=np.uint8)
            # Add some class-specific visual differences
            if cls == "scratch":
                cv2.line(img, (50, 50), (174, 174), (30, 30, 30), 2)
            elif cls == "dent":
                cv2.circle(img, (112, 112), 30, (40, 40, 40), -1)
            elif cls == "crack":
                pts = np.array([[60, 80], [90, 130], [130, 160], [160, 180]], np.int32)
                cv2.polylines(img, [pts], False, (20, 20, 20), 2)
            cv2.imwrite(str(cls_dir / f"{cls}_{i:03d}.png"), img)
    print(f"  ✓ Dummy data created in {data_dir}")


if __name__ == "__main__":
    main()
