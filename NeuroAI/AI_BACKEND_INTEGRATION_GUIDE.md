# 🧠 NeuroAI Diagnostic Engine — Full Backend & Frontend Integration Guide

> **Note for AI Models (ChatGPT, Claude, Gemini, etc.):**  
> This document is a complete technical specification of the **NeuroAI Multi-Stage Brain MRI Diagnostic System**. Use the details, endpoints, JSON schemas, and code templates provided below to generate backend integration code, API proxies, frontend components, or microservice wrappers in any language or framework requested by the developer.

---

## 1. System Overview & Architecture

NeuroAI is a specialized 4-stage sequential deep learning framework for clinical brain tumor localization, segmentation, classification, and explainability:

```
[Patient Brain MRI Upload (.jpg, .png, .h5)]
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: YOLOv8 Lesion Localization                    │
│ Model: yolo_best.pt (~22.5 MB)                         │
│ Output: Bounding Box Coordinates [x1, y1, x2, y2]      │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: SSL-Pretrained Swin-UNet Segmentation         │
│ Model: output_finetune/best_model.pth (~1.79 GB)       │
│ Output: Pixel-Level Lesion Boundary Margins (Mask)     │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: DenseNet-121 Multi-Specialist Classifier      │
│ Models: 4x densenet121_{class}.pth (~30.5 MB each)     │
│ Input: YOLO Contextual ROI (+15% margin crop)          │
│ Decision: One-vs-Rest Softmax (65% Confidence Gate)    │
│ Classes: GLIOMA, MENINGIOMA, PITUITARY, NOTUMOR,       │
│          UNRECOGNIZED_TUMOR (if top score < 65%)       │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Diagnostic Focal Heatmap (Grad-CAM)           │
│ Layer: DenseNet-121 features.denseblock4               │
│ Output: Saliency Heatmap Overlay & 5-Panel Display     │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
[JSON Result: Prediction, Confidence, Clinical Report, Base64 Images]
```

---

## 2. Server Specification & Port

- **Backend Framework:** Python Flask (`server.py`) with `flask-cors` enabled.
- **Default Server URL:** `http://127.0.0.1:8080` (or `http://localhost:8080`).
- **Start Command:**
  ```bash
  python server.py
  ```
- **Dependencies (`requirements.txt`):**
  ```text
  torch>=2.0.0
  torchvision>=0.15.0
  ultralytics>=8.0.0
  flask>=2.3.0
  flask-cors>=4.0.0
  opencv-python>=4.8.0
  numpy>=1.24.0
  pillow>=9.5.0
  matplotlib>=3.7.0
  h5py>=3.8.0
  timm>=0.9.0
  ```

---

## 3. REST API Specification

### Endpoint: `POST /api/predict`
Executes the full 4-stage diagnosis on a patient brain MRI image.

#### Request Headers:
```http
Content-Type: multipart/form-data
```

#### Request Body Options:
1. **Direct File Upload (Primary):**
   - Form field name: `file`
   - File types: `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.h5` (axial FLAIR brain MRI).
2. **Dataset Sample ID (For quick testing without uploading):**
   - Form field name: `sample_id`
   - Value: `"glioma"` | `"meningioma"` | `"pituitary"` | `"notumor"`

---

#### Response Format: `200 OK` (JSON)
```json
{
  "status": "success",
  "filename": "patient_mri.jpg",
  "pred_class": "GLIOMA",
  "conf_percent": 99.6,
  "segmented_pixels": 4532,
  "explanation_text": "=================================================================\n 🩺 CLINICAL DIAGNOSTIC REPORT\n=================================================================\n👨‍⚕️ FOR RADIOLOGISTS & CLINICIANS:\n  • Focal Heatmap Localization: High focal intensity across infiltrative, high-density cell structures in subcortical white matter.\n  • Tumor Boundary & Margin Delineation: Delineates irregular, ill-defined infiltrative boundary margins with surrounding peritumoral edema.\n  • Clinical Recommendation: Mass effect observed; recommend contrast-enhanced T1w and perfusion MRI for histopathological grading.\n\n👤 FOR PATIENTS & FAMILIES:\n  • What This Means: The scan identified an area of abnormal cell growth originating from supportive brain tissue (glial cells).\n  • Heatmap Explanation: The red outline and warm colored highlights show doctors the exact location where tissue structure differs from normal brain tissue.\n  • Suggested Next Steps: Discuss these findings with your neurologist or neurosurgeon to guide proper care and treatment planning.",
  "images": {
    "five_panel": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "bbox": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "seg": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "combined": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  }
}
```

#### Diagnostic Class Outcomes (`pred_class`):
- `GLIOMA`: High-confidence match with glioma pathology.
- `MENINGIOMA`: High-confidence match with extra-axial meningioma pathology.
- `PITUITARY`: High-confidence match with pituitary sellar mass pathology.
- `NOTUMOR`: Healthy brain tissue with symmetrical parenchyma and no lesion.
- `UNRECOGNIZED_TUMOR`: Abnormal focal lesion detected by YOLOv8 and Swin-UNet, but multi-specialist confidence is below the **65% threshold**. Requires urgent clinical biopsy or contrast MRI.

---

### Endpoint: `GET /api/samples`
Returns the built-in demo MRI test samples.

#### Response: `200 OK` (JSON)
```json
{
  "status": "success",
  "samples": [
    {"id": "glioma", "name": "Glioma MRI Sample", "class": "GLIOMA"},
    {"id": "meningioma", "name": "Meningioma MRI Sample", "class": "MENINGIOMA"},
    {"id": "pituitary", "name": "Pituitary MRI Sample", "class": "PITUITARY"},
    {"id": "notumor", "name": "Healthy Brain MRI Sample", "class": "NOTUMOR"}
  ]
}
```

---

## 4. How to Parse & Render the Results in the Frontend

### 1. Rendering the Images
All images are pre-encoded as standard **Base64 Data URLs**. You can render them directly in HTML / JSX without writing custom decoding logic:

```jsx
// React / JSX Example:
<img src={data.images.five_panel} alt="Complete 5-Panel Diagnostic Overview" style={{ width: '100%' }} />
```

### 2. Splitting the Medical Report (Radiologist vs. Patient)
The `explanation_text` string contains two clear sections separated by `👤 FOR PATIENTS & FAMILIES:`. Use this JavaScript helper to split them cleanly into separate UI tabs:

```javascript
function parseClinicalReport(fullText) {
  if (!fullText) return { clinicianReport: "", patientGuidance: "" };
  
  const delimiter = "👤 FOR PATIENTS & FAMILIES:";
  const parts = fullText.split(delimiter);
  
  return {
    clinicianReport: parts[0] ? parts[0].trim() : fullText,
    patientGuidance: parts[1] ? (delimiter + "\n" + parts[1].trim()) : fullText
  };
}

// Usage:
const { clinicianReport, patientGuidance } = parseClinicalReport(data.explanation_text);
```

---

## 5. Ready-to-Use Frontend Integration (React / Next.js)

```tsx
import React, { useState } from 'react';

interface DiagnosticResult {
  status: string;
  pred_class: string;
  conf_percent: number;
  segmented_pixels: number;
  explanation_text: string;
  images: {
    five_panel: string;
    bbox: string;
    seg: string;
    combined: string;
  };
}

export const NeuroAiUploader: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<DiagnosticResult | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'stages' | 'clinician' | 'patient'>('overview');

  const handleUpload = async () => {
    if (!file) return alert('Please select an MRI image first.');

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8080/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data: DiagnosticResult = await res.json();
      if (data.status === 'success') {
        setResult(data);
      } else {
        alert('Diagnostic Error: ' + (data as any).message);
      }
    } catch (err: any) {
      alert('Failed to connect to NeuroAI server: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getReportParts = () => {
    if (!result?.explanation_text) return { clinician: '', patient: '' };
    const parts = result.explanation_text.split('👤 FOR PATIENTS & FAMILIES:');
    return {
      clinician: parts[0]?.trim() || '',
      patient: parts[1] ? `👤 FOR PATIENTS & FAMILIES:\n${parts[1].trim()}` : '',
    };
  };

  const { clinician, patient } = getReportParts();

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20, fontFamily: 'sans-serif' }}>
      <h1>🧠 NeuroAI Diagnostic Portal</h1>

      {/* File Upload Input */}
      <div style={{ border: '2px dashed #3B82F6', padding: 30, textAlign: 'center', borderRadius: 12 }}>
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button onClick={handleUpload} disabled={loading || !file} style={{ marginLeft: 15, padding: '10px 20px', cursor: 'pointer' }}>
          {loading ? 'Processing AI Pipeline...' : 'Run MRI Diagnosis'}
        </button>
      </div>

      {/* Results Display */}
      {result && (
        <div style={{ marginTop: 30 }}>
          {/* Status Header */}
          <div style={{ padding: 15, borderRadius: 8, background: result.pred_class === 'UNRECOGNIZED_TUMOR' ? '#FEF2F2' : '#F0FDF4' }}>
            <h2>Diagnosis: {result.pred_class} ({result.conf_percent}%)</h2>
            <p>Segmented Lesion Area: {result.segmented_pixels} pixels</p>
          </div>

          {/* Navigation Tabs */}
          <div style={{ display: 'flex', gap: 10, margin: '20px 0' }}>
            <button onClick={() => setActiveTab('overview')}>📊 5-Panel Overview</button>
            <button onClick={() => setActiveTab('stages')}>📸 Diagnostic Stages</button>
            <button onClick={() => setActiveTab('clinician')}>👨‍⚕️ Clinician Report</button>
            <button onClick={() => setActiveTab('patient')}>👤 Patient Guidance</button>
          </div>

          {/* Tab Content */}
          {activeTab === 'overview' && (
            <img src={result.images.five_panel} alt="5-Panel Overview" style={{ width: '100%', borderRadius: 8 }} />
          )}

          {activeTab === 'stages' && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 15 }}>
              <div>
                <h4>Stage 1: Lesion Localization Box</h4>
                <img src={result.images.bbox} alt="Stage 1" style={{ width: '100%' }} />
              </div>
              <div>
                <h4>Stage 2: Margin Delineation</h4>
                <img src={result.images.seg} alt="Stage 2" style={{ width: '100%' }} />
              </div>
              <div>
                <h4>Stage 3: Combined Structural View</h4>
                <img src={result.images.combined} alt="Stage 3" style={{ width: '100%' }} />
              </div>
            </div>
          )}

          {activeTab === 'clinician' && (
            <pre style={{ background: '#1E293B', color: '#F8FAFC', padding: 20, borderRadius: 8, whiteSpace: 'pre-wrap' }}>
              {clinician}
            </pre>
          )}

          {activeTab === 'patient' && (
            <pre style={{ background: '#F8FAFC', border: '1px solid #CBD5E1', padding: 20, borderRadius: 8, whiteSpace: 'pre-wrap' }}>
              {patient}
            </pre>
          )}
        </div>
      )}
    </div>
  );
};
export default NeuroAiUploader;
```

---

## 6. Backend Integration Proxy (Node.js / Express Example)

If your architecture has an existing Node.js/Express backend that sits between the frontend and the Python AI service:

```javascript
// Express.js Route Proxy:
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

const NEUROAI_SERVICE_URL = process.env.NEUROAI_SERVICE_URL || 'http://127.0.0.1:8080/api/predict';

router.post('/diagnose', upload.single('mri_file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No file uploaded.' });
    }

    // Forward the file as multipart/form-data to the Python AI engine
    const form = new FormData();
    form.append('file', req.file.buffer, req.file.originalname);

    const aiResponse = await axios.post(NEUROAI_SERVICE_URL, form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    return res.status(200).json({
      success: true,
      data: aiResponse.data,
    });
  } catch (error) {
    console.error('NeuroAI Proxy Error:', error.message);
    return res.status(500).json({
      success: false,
      message: 'Failed to communicate with NeuroAI engine.',
      error: error.message,
    });
  }
});

module.exports = router;
```

---

## 7. Model Weights Checklist

Ensure the following 6 model weight files exist in their specified relative locations:

| Relative File Path | Size | Description |
| :--- | :--- | :--- |
| `yolo_best.pt` | ~22.5 MB | YOLOv8 Lesion Localization |
| `output_finetune/best_model.pth` | ~1.79 GB | Swin-UNet Pixel Segmentation |
| `densenet121_glioma.pth` | ~30.5 MB | Glioma Specialist Classifier |
| `densenet121_meningioma.pth` | ~30.5 MB | Meningioma Specialist Classifier |
| `densenet121_pituitary.pth` | ~30.5 MB | Pituitary Specialist Classifier |
| `densenet121_notumor.pth` | ~30.5 MB | Normal/Healthy Brain Classifier |

---

## 8. Summary Checklist for Other AI Models
When asking another AI model to integrate this system into your app:
1. **Pass this `.md` file directly** to the AI model.
2. Ask it to generate the specific client in your preferred language (e.g., *"Write a Next.js Server Action that calls the `/api/predict` endpoint defined in this guide"*, or *"Create a FastAPI microservice wrapper around `Neuro_AI_System.py`"*).
3. The AI model will have all endpoint signatures, payload parameters, Base64 image formats, and report parsing logic needed to generate code.
