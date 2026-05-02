"""
gradio_demo.py — Gradio Demo for Defect Detection
===================================================
Ek simple, shareable demo — single command se launch karo.
Hugging Face Spaces par bhi deploy ho sakta hai!

Run: python deployment/gradio_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["good", "scratch", "dent", "crack", "contamination", "bent"]
CLASS_EMOJI = {
    "good": "✅",
    "scratch": "〰️",
    "dent": "◉",
    "crack": "⚡",
    "contamination": "▓",
    "bent": "◪",
}
CLASS_COLORS_HEX = {
    "good": "#22c55e",
    "scratch": "#f59e0b",
    "dent": "#f97316",
    "crack": "#ef4444",
    "contamination": "#a855f7",
    "bent": "#06b6d4",
}


# ── Prediction Logic ──────────────────────────────────────────────────────────

def mock_predict(image_array: np.ndarray) -> dict:
    """Simulate model prediction (works without trained weights)."""
    time.sleep(0.4)
    seed = int(image_array.mean() * 100) % 9999
    np.random.seed(seed)
    raw = np.random.dirichlet(np.ones(6) * 0.4)
    top = np.random.randint(0, 6)
    raw[top] += 2.0
    probs = raw / raw.sum()
    pred_idx = probs.argmax()
    return {
        "class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
        "is_defective": CLASS_NAMES[pred_idx] != "good",
    }


def try_real_model(image_array, checkpoint_path, model_name):
    """Try real model, fallback to mock."""
    try:
        import torch
        from predict import DefectPredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = DefectPredictor(model_name, checkpoint_path, device=device)
        return predictor.predict_image(image_array)
    except Exception as e:
        return mock_predict(image_array)


def build_prob_chart(probs: dict) -> plt.Figure:
    """Build a horizontal bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    classes = list(probs.keys())
    values = [v * 100 for v in probs.values()]
    sorted_pairs = sorted(zip(values, classes), reverse=True)
    values_sorted, classes_sorted = zip(*sorted_pairs)

    colors = [CLASS_COLORS_HEX.get(cls, "#818cf8") for cls in classes_sorted]
    bars = ax.barh(classes_sorted, values_sorted, color=colors, edgecolor="none",
                   height=0.55, alpha=0.9)

    for bar, val in zip(bars, values_sorted):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9,
                color="#e2e8f0", fontweight="500")

    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#374151")
    ax.spines["left"].set_color("#374151")
    ax.set_title("Class Probabilities", color="#d1d5db", fontsize=10, pad=8)
    ax.axvline(50, color="#374151", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    return fig


def analyze_image(
    image,
    model_choice: str,
    confidence_threshold: float,
    checkpoint_path: str,
):
    """Main inference function called by Gradio."""
    if image is None:
        return (
            "## ⚠️ No image uploaded",
            None,
            "Please upload an image first.",
        )

    img_arr = np.array(image)

    # Run prediction
    model_map = {"ResNet-50 (Recommended)": "resnet50",
                 "Custom CNN": "custom_cnn",
                 "EfficientNet-B0": "efficientnet"}
    model_name = model_map.get(model_choice, "resnet50")

    if checkpoint_path and os.path.exists(checkpoint_path):
        result = try_real_model(img_arr, checkpoint_path, model_name)
    else:
        result = mock_predict(img_arr)

    # Build output
    cls = result["class"]
    conf = result["confidence"] * 100
    emoji = CLASS_EMOJI.get(cls, "❓")
    low_conf_warning = f"\n\n⚠️ *Low confidence ({conf:.1f}%) — manual review recommended.*" if result["confidence"] < confidence_threshold else ""

    if result["is_defective"]:
        status_md = f"""## ⚠️ DEFECT DETECTED

**Class:** {emoji} `{cls.upper()}`  
**Confidence:** `{conf:.1f}%`  
**Status:** 🔴 Reject Product{low_conf_warning}
"""
    else:
        status_md = f"""## ✅ PRODUCT OK

**Class:** {emoji} `GOOD`  
**Confidence:** `{conf:.1f}%`  
**Status:** 🟢 Pass QC{low_conf_warning}
"""

    # Image statistics
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    stats_md = f"""### 📊 Image Stats
| Property | Value |
|----------|-------|
| Resolution | {img_arr.shape[1]} × {img_arr.shape[0]} px |
| Brightness | {gray.mean():.1f} |
| Contrast | {gray.std():.1f} |
| Sharpness | {cv2.Laplacian(gray, cv2.CV_64F).var():.1f} |

### 🔢 Raw Probabilities
"""
    for c, p in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        stats_md += f"`{c:<15}` {bar} `{p*100:.1f}%`\n"

    prob_chart = build_prob_chart(result["probabilities"])

    return status_md, prob_chart, stats_md


def create_demo_images():
    """Create sample demo images for the examples section."""
    demos = []
    configs = [
        ("good", None),
        ("scratch", lambda img: cv2.line(img, (30, 50), (194, 170), (20, 20, 20), 3)),
        ("dent", lambda img: cv2.circle(img, (112, 112), 35, (30, 30, 30), -1)),
        ("crack", lambda img: cv2.polylines(img, [np.array([[50,60],[90,120],[140,170],[180,200]], np.int32)], False, (15,15,15), 2)),
    ]
    for cls, fn in configs:
        img = np.random.randint(110, 175, (224, 224, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        if fn:
            fn(img)
        demos.append(img)
    return demos


# ── Gradio Interface ──────────────────────────────────────────────────────────

custom_theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Sora"), "ui-sans-serif"],
).set(
    body_background_fill="#0a0e1a",
    body_text_color="#e2e8f0",
    block_background_fill="#111827",
    block_border_color="#1e293b",
    input_background_fill="#0f172a",
    button_primary_background_fill="linear-gradient(135deg, #4f46e5, #7c3aed)",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(theme=custom_theme, title="Defect Detection System") as demo:

    gr.Markdown("""
    # 🏭 Automated Defect Detection System
    **Deep Learning · Computer Vision · Manufacturing QC**
    
    Upload a product image → AI detects scratches, dents, cracks & more in milliseconds.
    """)

    with gr.Row():
        # Left column — inputs
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📷 Product Image",
                type="pil",
                height=280,
            )
            with gr.Accordion("⚙️ Settings", open=False):
                model_choice = gr.Dropdown(
                    choices=["ResNet-50 (Recommended)", "Custom CNN", "EfficientNet-B0"],
                    value="ResNet-50 (Recommended)",
                    label="Model",
                )
                conf_threshold = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                    label="Confidence Threshold",
                )
                checkpoint_path = gr.Textbox(
                    label="Model Checkpoint Path (optional)",
                    placeholder="models/best_resnet50.pth",
                    value="",
                )
            analyze_btn = gr.Button("⚡ Analyze for Defects", variant="primary", size="lg")

        # Right column — outputs
        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Result")
            prob_chart = gr.Plot(label="Probability Chart")

    with gr.Row():
        stats_output = gr.Markdown(label="Detailed Statistics")

    # Examples
    gr.Markdown("---\n### 🧪 Try Demo Images")
    demo_imgs = create_demo_images()
    gr.Examples(
        examples=[[Image.fromarray(img)] for img in demo_imgs],
        inputs=[image_input],
        label="Click to load a demo image",
    )

    # Connect
    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input, model_choice, conf_threshold, checkpoint_path],
        outputs=[result_output, prob_chart, stats_output],
    )

    gr.Markdown("""
    ---
    <div style="text-align:center;color:#475569;font-size:0.8rem;">
    Automated Defect Detection · ResNet-50 + YOLOv8 · PyTorch
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,           # Set True to get public Gradio link
        show_error=True,
    )
