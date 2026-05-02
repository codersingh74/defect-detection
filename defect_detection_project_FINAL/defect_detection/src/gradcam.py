"""
gradcam.py — GradCAM Visualization for Explainability
======================================================
Shows WHERE the model is looking when classifying defects.
Run: python src/gradcam.py --image path/to/image.png --checkpoint models/best_resnet50.pth
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from dataset import CLASS_NAMES, get_val_transforms
from models import get_model, load_checkpoint


# ── GradCAM Implementation ────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM).
    Visualizes which regions of the image the model focuses on.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: (1, C, H, W) preprocessed image
            class_idx: Target class (None = predicted class)

        Returns:
            cam: (H, W) numpy array, values in [0, 1]
            pred_class: Predicted class index
            pred_conf: Prediction confidence
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_conf = probs[0, pred_class].item()

        if class_idx is None:
            class_idx = pred_class

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Compute GradCAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pool
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive influence

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, pred_class, pred_conf


def apply_heatmap(cam: np.ndarray, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay GradCAM heatmap on original image."""
    # Resize CAM to image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    return overlay, heatmap


def visualize_gradcam(
    image_path: str,
    model: torch.nn.Module,
    cam_layer: torch.nn.Module,
    class_names: list,
    save_path: str = None,
    device: str = "cpu",
):
    """Generate and display GradCAM visualization for a single image."""
    # Load and preprocess image
    img_orig = cv2.imread(image_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_orig, (224, 224))

    transform = get_val_transforms(224)
    img_tensor = transform(image=img_resized)["image"].unsqueeze(0).to(device)

    # Generate GradCAM
    gradcam = GradCAM(model, cam_layer)
    cam, pred_class, pred_conf = gradcam.generate_cam(img_tensor)

    # Apply heatmap
    overlay, heatmap = apply_heatmap(cam, img_resized)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"GradCAM — Predicted: {class_names[pred_class]} ({pred_conf*100:.1f}%)",
        fontsize=13, fontweight="bold"
    )

    titles = ["Original", "GradCAM Heatmap", "Overlay", "CAM (grayscale)"]
    images = [img_resized, heatmap, overlay, (cam * 255).astype(np.uint8)]
    cmaps = [None, None, None, "hot"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # Add class probability bars
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    ax_bar = fig.add_axes([0.92, 0.15, 0.06, 0.7])
    colors = ["#185FA5" if i == pred_class else "#B5D4F4" for i in range(len(class_names))]
    ax_bar.barh(class_names, probs * 100, color=colors)
    ax_bar.set_xlabel("Prob (%)", fontsize=8)
    ax_bar.tick_params(labelsize=7)
    ax_bar.set_xlim(0, 110)

    plt.tight_layout(rect=[0, 0, 0.91, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ GradCAM saved to: {save_path}")
    else:
        plt.show()

    plt.close()
    return pred_class, pred_conf


def batch_gradcam(
    image_dir: str,
    model: torch.nn.Module,
    cam_layer: torch.nn.Module,
    class_names: list,
    save_dir: str,
    n_images: int = 8,
    device: str = "cpu",
):
    """Generate GradCAM for a batch of images in a grid layout."""
    image_dir = Path(image_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    images = list(image_dir.rglob("*.png"))[:n_images]
    if not images:
        print("No images found!")
        return

    n_cols = 4
    n_rows = (len(images) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows))
    fig.suptitle("GradCAM — Batch Visualization", fontsize=14, fontweight="bold")

    transform = get_val_transforms(224)
    gradcam = GradCAM(model, cam_layer)

    for idx, img_path in enumerate(images):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))

        img_tensor = transform(image=img_resized)["image"].unsqueeze(0).to(device)
        cam, pred_class, pred_conf = gradcam.generate_cam(img_tensor)
        overlay, _ = apply_heatmap(cam, img_resized)

        ax.imshow(overlay)
        ax.set_title(f"{class_names[pred_class]}\n{pred_conf*100:.1f}%", fontsize=9)
        ax.axis("off")

    # Hide empty subplots
    for idx in range(len(images), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "gradcam_batch.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Batch GradCAM saved to: {save_dir}/gradcam_batch.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GradCAM Visualization")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to image directory (batch mode)")
    parser.add_argument("--checkpoint", type=str, default="models/best_resnet50.pth")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--save_dir", type=str, default="results/plots")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    model = get_model(args.model, args.num_classes, device=device)
    model = load_checkpoint(model, args.checkpoint, device)

    # Get target layer
    cam_layer = model.backbone.layer4[-1] if hasattr(model, "backbone") else list(model.modules())[-3]

    if args.image:
        save_path = Path(args.save_dir) / "gradcam_single.png"
        pred_class, pred_conf = visualize_gradcam(
            args.image, model, cam_layer, CLASS_NAMES, str(save_path), device
        )
        print(f"\nPrediction: {CLASS_NAMES[pred_class]} ({pred_conf*100:.1f}% confidence)")
    elif args.image_dir:
        batch_gradcam(args.image_dir, model, cam_layer, CLASS_NAMES, args.save_dir, device=device)
    else:
        print("Provide --image or --image_dir")


if __name__ == "__main__":
    main()
