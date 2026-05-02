"""
dataset.py — Dataset Classes & DataLoader Utilities
====================================================
PyTorch Dataset for defect detection — supports both
classification (ResNet) and detection (YOLO) formats.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


# ── Class Labels ─────────────────────────────────────────────────────────────

CLASS_NAMES = ["good", "scratch", "dent", "crack", "contamination", "bent"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}


# ── Augmentation Pipelines ───────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Heavy augmentation for training — reduces overfitting."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Minimal transforms for validation/test — only resize + normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Dataset Class ────────────────────────────────────────────────────────────

class DefectDataset(Dataset):
    """
    PyTorch Dataset for defect classification.

    Directory structure expected:
        data_dir/
            good/      image1.png, image2.png ...
            scratch/   image1.png ...
            dent/      ...
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[A.Compose] = None,
        split: str = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.samples: List[Tuple[str, int]] = []

        self._load_samples(train_ratio, val_ratio, seed)

    def _load_samples(self, train_ratio, val_ratio, seed):
        """Load image paths and labels, then split into train/val/test."""
        np.random.seed(seed)
        all_samples = []

        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[class_name]
            images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            all_samples.extend([(str(p), label) for p in images])

        # Shuffle and split
        indices = np.random.permutation(len(all_samples))
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train:n_train + n_val]
        else:  # test
            selected = indices[n_train + n_val:]

        self.samples = [all_samples[i] for i in selected]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        labels = [lbl for _, lbl in self.samples]
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        counts = np.maximum(counts, 1)  # avoid division by zero
        weights = 1.0 / counts
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        return [class_weights[lbl] for _, lbl in self.samples]


# ── DataLoader Factory ───────────────────────────────────────────────────────

def create_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Creates train / val / test DataLoaders.

    Args:
        data_dir: Root directory with class subdirectories.
        image_size: Resize target (224 for ResNet, 640 for YOLO).
        batch_size: Mini-batch size.
        num_workers: CPU workers for data loading.
        use_weighted_sampler: Handle class imbalance via sampling.

    Returns:
        Dict with keys "train", "val", "test".
    """
    datasets = {
        "train": DefectDataset(data_dir, get_train_transforms(image_size), split="train"),
        "val": DefectDataset(data_dir, get_val_transforms(image_size), split="val"),
        "test": DefectDataset(data_dir, get_val_transforms(image_size), split="test"),
    }

    # Weighted sampler for training to handle class imbalance
    train_sampler = None
    if use_weighted_sampler and len(datasets["train"]) > 0:
        sample_weights = datasets["train"].get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    # Print dataset info
    for split, ds in datasets.items():
        print(f"  {split:>5}: {len(ds):>5} images")

    return loaders
