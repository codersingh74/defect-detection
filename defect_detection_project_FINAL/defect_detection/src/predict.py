"""
predict.py — Inference Script
==============================
Run defect detection on single images, folders, or live webcam.
Run: python src/predict.py --source path/to/image.jpg --checkpoint models/best_resnet50.pth
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from dataset import CLASS_NAMES, get_val_transforms
from models import get_model, load_checkpoint


# ── Inference Functions ───────────────────────────────────────────────────────

class DefectPredictor:
    """High-level inference class for defect detection."""

    def __init__(self, model_name: str, checkpoint: str, num_classes: int = 6, device: str = "cpu"):
        self.device = device
        self.class_names = CLASS_NAMES[:num_classes]
        self.transform = get_val_transforms(224)

        self.model = get_model(model_name, num_classes, device=device)
        self.model = load_checkpoint(self.model, checkpoint, device)
        self.model.eval()
        print(f"✓ Model ready for inference on {device.upper()}")

    @torch.no_grad()
    def predict_image(self, image: np.ndarray) -> dict:
        """
        Predict defect class for a single image (numpy RGB array).

        Returns:
            {
                "class": "scratch",
                "confidence": 0.94,
                "probabilities": {"good": 0.01, "scratch": 0.94, ...}
            }
        """
        img_resized = cv2.resize(image, (224, 224))
        tensor = self.transform(image=img_resized)["image"].unsqueeze(0).to(self.device)

        output = self.model(tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

        pred_idx = probs.argmax()
        return {
            "class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {cls: float(p) for cls, p in zip(self.class_names, probs)},
            "is_defective": self.class_names[pred_idx] != "good",
        }

    def predict_file(self, img_path: str) -> dict:
        """Load image from file and predict."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.predict_image(img_rgb)
        result["path"] = img_path
        return result

    def predict_folder(self, folder: str, extensions=(".png", ".jpg", ".jpeg")):
        """Predict all images in a folder. Returns list of results."""
        folder = Path(folder)
        images = [p for ext in extensions for p in folder.rglob(f"*{ext}")]
        print(f"Found {len(images)} images in {folder}")

        results = []
        for img_path in images:
            try:
                r = self.predict_file(str(img_path))
                results.append(r)
                status = "⚠️  DEFECT" if r["is_defective"] else "✅ GOOD"
                print(f"  {status} [{r['class']:>15}] {r['confidence']*100:>5.1f}%  {img_path.name}")
            except Exception as e:
                print(f"  ❌ Error: {img_path.name} — {e}")
        return results

    def predict_live(self, camera_id: int = 0):
        """
        Real-time inference from webcam / USB camera.
        Press 'q' to quit, 's' to save screenshot.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return

        print("🎥 Live inference started. Press 'q' to quit, 's' to save.")
        frame_count = 0
        save_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Predict every 5 frames to save CPU
            frame_count += 1
            if frame_count % 5 == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.predict_image(img_rgb)

                # Draw overlay
                label = result["class"]
                conf = result["confidence"]
                color = (0, 200, 50) if not result["is_defective"] else (0, 50, 220)

                cv2.rectangle(frame, (10, 10), (320, 70), (0, 0, 0), -1)
                cv2.putText(frame, f"Class: {label}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Confidence: {conf*100:.1f}%", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Defect Detection QC System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_count += 1
                cv2.imwrite(f"screenshot_{save_count:03d}.jpg", frame)
                print(f"📸 Saved screenshot_{save_count:03d}.jpg")

        cap.release()
        cv2.destroyAllWindows()


def visualize_prediction(img_path: str, result: dict, save_path: str = None):
    """Visualize prediction with probability bars."""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_disp = cv2.resize(img_rgb, (224, 224))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Prediction: {result['class'].upper()} ({result['confidence']*100:.1f}%)",
        fontsize=13, fontweight="bold",
        color="#993556" if result["is_defective"] else "#0F6E56"
    )

    axes[0].imshow(img_disp)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    probs = result["probabilities"]
    classes = list(probs.keys())
    values = [v * 100 for v in probs.values()]
    colors = ["#185FA5" if cls == result["class"] else "#B5D4F4" for cls in classes]

    bars = axes[1].barh(classes, values, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_title("Class Probabilities")
    axes[1].set_xlabel("Probability (%)")
    axes[1].set_xlim(0, 115)
    axes[1].axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar, val in zip(bars, values):
        axes[1].text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Visualization saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Defect Detection Inference")
    parser.add_argument("--source", type=str, default=None,
                        help="Image path, folder path, or 0 for webcam")
    parser.add_argument("--checkpoint", type=str, default="models/best_resnet50.pth")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    predictor = DefectPredictor(args.model, args.checkpoint, args.num_classes, device)

    if args.source is None:
        print("Usage: python predict.py --source <image|folder|0>")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "0":
        predictor.predict_live(camera_id=0)
    elif Path(args.source).is_dir():
        results = predictor.predict_folder(args.source)
        defective = sum(1 for r in results if r["is_defective"])
        print(f"\n📊 Summary: {defective}/{len(results)} defective ({defective/len(results)*100:.1f}%)")
    elif Path(args.source).is_file():
        result = predictor.predict_file(args.source)
        print(f"\n  Class      : {result['class']}")
        print(f"  Confidence : {result['confidence']*100:.1f}%")
        print(f"  Defective  : {'Yes ⚠️' if result['is_defective'] else 'No ✅'}")
        visualize_prediction(args.source, result,
                             save_path=str(save_dir / "prediction_result.png"))
    else:
        print(f"❌ Source not found: {args.source}")


if __name__ == "__main__":
    main()
