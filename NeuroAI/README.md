# Neuro AI Diagnostic System

This module contains the standalone machine learning inference engine, neural network architectures, model weights management, and REST API server for the Brain Tumor Detection System.

---

## Architectural Overview

The Neuro AI engine integrates four deep learning and computer vision stages:

1. **Stage 1 — YOLOv8 Object Detection**: Localizes abnormal brain tissue and extracts bounding box spatial coordinates.
2. **Stage 2 — Swin-UNet Segmentation**: Evaluates a Hierarchical Vision Transformer to delineate pixel-level tumor boundaries and calculate segmented lesion area.
3. **Stage 3 — DenseNet-121 Multi-Specialist Classification**: Evaluates four binary specialist classifiers (Glioma, Meningioma, Pituitary, Normal/No-Tumor) on background-blurred Regions of Interest (ROI).
4. **Stage 4 — Grad-CAM Explainable AI (XAI)**: Generates class activation gradient maps and compiles a 5-panel clinical visual summary.

---

## Directory & File Structure

### 1. API Server (`server.py`)
- **Type**: Flask REST Service
- **Address**: `http://127.0.0.1:8080`
- **Endpoints**:
  - `POST /api/predict`: Accepts MRI images (`.png`, `.jpg`, `.jpeg`, `.h5`), runs the complete diagnostic pipeline, and returns predictions, confidence percentages, segmented pixel metrics, dual clinical text reports, and Base64-encoded output images.
  - `GET /api/samples`: Returns pre-configured reference scan samples for verification.

### 2. Core Inference Engine (`Neuro_AI_System.py`)
- Coordinates image ingestion, preprocessing (224x224 normalization), model evaluation, Grad-CAM heatmap extraction, and clinical text synthesis.
- Implements atypical lesion handling when abnormal tissue is segmented but specialist confidence is below standard classification thresholds.

### 3. Swin-UNet Architecture (`extracted_pipeline/`)
- Contains PyTorch implementations of Shifted Window Multi-Head Self-Attention (`W-MSA` / `SW-MSA`), Swin-UNet decoder layers, and YAML hyperparameter configurations (`configs/swin_base_patch4_window7_224_finetune.yaml`).

### 4. Model Checkpoints (Production Weights)

| Model File | Target Role | Size | Status |
| :--- | :--- | :--- | :--- |
| `output_finetune/best_model.pth` (or `best_model.pth`) | Swin-UNet Semantic Segmentation | ~1.79 GB | Installed Locally |
| `yolo_best.pt` | YOLOv8 Bounding Box Localization | ~22.5 MB | Installed Locally |
| `densenet121_glioma.pth` | Glioma Specialist Classifier | ~30.5 MB | Installed Locally |
| `densenet121_meningioma.pth` | Meningioma Specialist Classifier | ~30.5 MB | Installed Locally |
| `densenet121_pituitary.pth` | Pituitary Specialist Classifier | ~30.5 MB | Installed Locally |
| `densenet121_notumor.pth` | Normal Tissue Specialist Classifier | ~30.5 MB | Installed Locally |

*See `AI_BACKEND_INTEGRATION_GUIDE.md` for full technical specifications, JSON response contracts, Grad-CAM details, and API proxies.*

---

## Execution Instructions

### Running the REST API Server

```bash
python server.py
```

### Running the Standalone Streamlit Interface

```bash
streamlit run app.py
```

---

## API Reference

### `POST /api/predict`

**Request**:
- Content-Type: `multipart/form-data`
- Body Parameter: `file` (Binary Image) or `sample_id` (String)

**Sample Response**:

```json
{
  "status": "success",
  "filename": "mri_scan.jpg",
  "pred_class": "MENINGIOMA",
  "conf_percent": 97.04,
  "segmented_pixels": 15416,
  "explanation_text": "...",
  "images": {
    "bbox": "data:image/png;base64,...",
    "seg": "data:image/png;base64,...",
    "combined": "data:image/png;base64,...",
    "five_panel": "data:image/png;base64,..."
  }
}
```
