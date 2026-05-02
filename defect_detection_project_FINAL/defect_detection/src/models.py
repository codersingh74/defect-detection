"""
models.py — Model Definitions
==============================
1. CustomCNN     — Baseline CNN from scratch
2. ResNet50Model — Transfer Learning with pretrained ResNet-50
3. get_model()   — Factory function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


# ── 1. Custom CNN (Baseline) ──────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool block."""
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Baseline CNN for defect classification.

    Architecture:
        Input (3×224×224)
        → ConvBlock(32)  → 32×112×112
        → ConvBlock(64)  → 64×56×56
        → ConvBlock(128) → 128×28×28
        → ConvBlock(256) → 256×14×14
        → GlobalAvgPool  → 256
        → FC(512) → Dropout → FC(num_classes)
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)


# ── 2. ResNet-50 Transfer Learning ───────────────────────────────────────────

class ResNet50Model(nn.Module):
    """
    ResNet-50 fine-tuned for defect detection.

    Strategy:
        - Load ImageNet pretrained weights
        - Freeze all layers except last 20 parameters
        - Replace final FC layer with custom classification head
    """

    def __init__(self, num_classes: int = 6, dropout: float = 0.4, freeze_layers: bool = True):
        super().__init__()

        # Load pretrained backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)

        # Freeze early layers
        if freeze_layers:
            params = list(backbone.parameters())
            for param in params[:-20]:
                param.requires_grad = False

        # Replace classifier head
        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()             # Remove original FC
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def unfreeze_all(self):
        """Call after initial training to fine-tune all layers."""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ All layers unfrozen for full fine-tuning")

    def get_cam_layer(self):
        """Return last conv layer for GradCAM."""
        return self.backbone.layer4[-1]


# ── 3. EfficientNet Option ────────────────────────────────────────────────────

class EfficientNetModel(nn.Module):
    """EfficientNet-B0 — lightweight alternative to ResNet."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)

        # Freeze feature extractor
        for param in backbone.features.parameters():
            param.requires_grad = False

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── 4. Factory Function ───────────────────────────────────────────────────────

def get_model(
    model_name: str = "resnet50",
    num_classes: int = 6,
    dropout: float = 0.4,
    freeze_layers: bool = True,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        model_name: "custom_cnn" | "resnet50" | "efficientnet"
        num_classes: Number of output classes.
        dropout: Dropout rate for classifier.
        freeze_layers: Freeze backbone layers (for transfer learning).
        device: "cuda" or "cpu". Auto-detected if None.

    Returns:
        Initialized model on the specified device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_map = {
        "custom_cnn": lambda: CustomCNN(num_classes=num_classes),
        "resnet50": lambda: ResNet50Model(num_classes=num_classes, dropout=dropout, freeze_layers=freeze_layers),
        "efficientnet": lambda: EfficientNetModel(num_classes=num_classes, dropout=dropout),
    }

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")

    model = model_map[model_name]().to(device)

    # Count trainable parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model      : {model_name}")
    print(f"  Parameters : {total:,} total | {trainable:,} trainable")
    print(f"  Device     : {device}")

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load model weights from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Accuracy: {checkpoint.get('val_acc', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    return model.to(device)
