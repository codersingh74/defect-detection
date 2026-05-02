# 🏭 Automated Defect Detection System
### Deep Learning + Computer Vision for Manufacturing QC

---

## 📌 Project Overview

Yeh project manufacturing industries ke liye ek **AI-powered Quality Control System** hai jo factory belt par aane wale products mein defects (scratches, dents, cracks, contamination) automatically detect karta hai.

**Models Used:**
- 🔵 **Custom CNN** — Baseline binary classifier
- 🟢 **ResNet-50** — Transfer Learning se fine-tuned classifier
- 🟠 **YOLOv8** — Real-time object detection with bounding boxes

---

## 📁 Project Structure

```
defect_detection/
├── data/
│   ├── raw/                   # Original images (download here)
│   ├── processed/             # Preprocessed images
│   └── annotations/           # YOLO format labels (.txt)
├── src/
│   ├── dataset.py             # Dataset class & DataLoader
│   ├── models.py              # CNN, ResNet-50, model utilities
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Metrics, confusion matrix
│   ├── predict.py             # Inference on new images
│   ├── eda.py                 # Exploratory Data Analysis
│   ├── augmentation.py        # Albumentations pipeline
│   └── gradcam.py             # GradCAM explainability
├── notebooks/
│   ├── 01_EDA.ipynb           # Data analysis notebook
│   ├── 02_Baseline_CNN.ipynb  # CNN training walkthrough
│   ├── 03_ResNet50.ipynb      # Transfer learning notebook
│   └── 04_YOLOv8.ipynb        # YOLO training & inference
├── models/                    # Saved model weights (.pth, .pt)
├── configs/
│   ├── config.yaml            # Main configuration
│   └── dataset.yaml           # YOLO dataset config
├── deployment/
│   ├── app.py                 # Streamlit dashboard
│   └── gradio_demo.py         # Gradio quick demo
├── results/
│   ├── plots/                 # Training curves, confusion matrix
│   └── weights/               # Best model checkpoints
├── requirements.txt           # Dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
git clone <your-repo>
cd defect_detection
pip install -r requirements.txt
```

### 2. Dataset Download
```bash
# MVTec Anomaly Detection Dataset (Recommended)
# Download from: https://www.mvtec.com/company/research/datasets/mvtec-ad
# Place in: data/raw/

# Ya Kaggle se:
kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database
```

### 3. EDA Run Karo
```bash
python src/eda.py --data_dir data/raw/
```

### 4. ResNet-50 Train Karo
```bash
python src/train.py --model resnet50 --epochs 50 --batch_size 32
```

### 5. YOLOv8 Train Karo
```bash
python src/train.py --model yolov8 --epochs 100
```

### 6. Dashboard Launch Karo
```bash
streamlit run deployment/app.py
```

---

## 📊 Results (Expected)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN | ~82% | 0.80 | 0.79 | 0.79 |
| ResNet-50 | ~93% | 0.92 | 0.91 | 0.91 |
| YOLOv8 (mAP@50) | ~89% | 0.91 | 0.88 | 0.89 |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch 2.x, Ultralytics YOLOv8 |
| Computer Vision | OpenCV, Albumentations |
| Data Analysis | NumPy, Pandas, Matplotlib, Seaborn |
| Explainability | GradCAM (pytorch-grad-cam) |
| Deployment | Streamlit, Gradio |
| Utilities | tqdm, scikit-learn, Pillow |

---

## 🎯 Defect Classes

1. **Scratch** — Surface par thin linear marks
2. **Dent** — Circular/irregular surface deformation  
3. **Crack** — Fracture lines in material
4. **Contamination** — Foreign particles ya stains
5. **Bent** — Shape deformation / warping
6. **Good** — Defect-free (Normal class)

---

## 📈 Key Features

- ✅ Deep EDA with statistical analysis
- ✅ Class imbalance handling (WeightedRandomSampler)
- ✅ Transfer Learning with ResNet-50
- ✅ Real-time detection with YOLOv8
- ✅ GradCAM heatmaps (explainability)
- ✅ Streamlit dashboard for demo
- ✅ Modular, production-ready code

---

## 👤 Author

**Your Name**  
Deep Learning | Computer Vision | Data Science

---
