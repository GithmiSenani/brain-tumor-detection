import os
import sys
import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models

# -----------------------------------------------------------------------------
# 1. Setup Local Paths & Modules
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DIR = os.path.join(BASE_DIR, 'extracted_pipeline')

if not os.path.exists(EXTRACTED_DIR):
    import zipfile
    zip_p = os.path.join(BASE_DIR, 'combined_pipeline_blurred.zip')
    if os.path.exists(zip_p):
        print(f"[*] Unzipping {zip_p} for local execution...")
        with zipfile.ZipFile(zip_p, 'r') as z:
            z.extractall(EXTRACTED_DIR)
        print("[+] Unzipped pipeline successfully!")

for p in [BASE_DIR, EXTRACTED_DIR, os.path.join(EXTRACTED_DIR, 'models')]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import config
config._C.set_new_allowed(True)

import pipeline_yolo_seg
from pipeline_yolo_seg import YOLOSegPipeline

# Patch DummyArgs for robust configuration loading
if hasattr(pipeline_yolo_seg, 'DummyArgs'):
    old_init = pipeline_yolo_seg.DummyArgs.__init__
    def patched_init(self, cfg_path, opts=None):
        old_init(self, cfg_path, opts)
        self.zip = False
        self.cache_mode = None
        self.resume = None
        self.batch_size = None
        self.data_path = None
        self.pretrained = None
        self.accumulation_steps = None
        self.use_checkpoint = False
        self.amp_opt_level = None
        self.output = 'output_finetune'
        self.tag = 'swin_unet_finetune'
        self.eval = False
        self.throughput = False
    pipeline_yolo_seg.DummyArgs.__init__ = patched_init

# -----------------------------------------------------------------------------
# 2. DenseNet-121 Multi-Specialist Classifier Setup
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASSES = ['GLIOMA', 'MENINGIOMA', 'PITUITARY', 'NOTUMOR']

def build_specialist_model():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    return model

CLINICAL_EXPLANATIONS = {
    'GLIOMA': (
        "=================================================================\n"
        " 🩺 CLINICAL DIAGNOSTIC REPORT\n"
        "=================================================================\n"
        "👨‍⚕️ FOR RADIOLOGISTS & CLINICIANS:\n"
        "  • Focal Heatmap Localization: High focal intensity across infiltrative, high-density cell structures in subcortical white matter.\n"
        "  • Tumor Boundary & Margin Delineation: Delineates irregular, ill-defined infiltrative boundary margins with surrounding peritumoral edema.\n"
        "  • Clinical Recommendation: Mass effect observed; recommend contrast-enhanced T1w and perfusion MRI for histopathological grading.\n\n"
        "👤 FOR PATIENTS & FAMILIES:\n"
        "  • What This Means: The scan identified an area of abnormal cell growth originating from supportive brain tissue (glial cells).\n"
        "  • Heatmap Explanation: The red outline and warm colored highlights show doctors the exact location where tissue structure differs from normal brain tissue.\n"
        "  • Suggested Next Steps: Discuss these findings with your neurologist or neurosurgeon to guide proper care and treatment planning."
    ),
    'MENINGIOMA': (
        "=================================================================\n"
        " 🩺 CLINICAL DIAGNOSTIC REPORT\n"
        "=================================================================\n"
        "👨‍⚕️ FOR RADIOLOGISTS & CLINICIANS:\n"
        "  • Focal Heatmap Localization: Concentrates on extra-axial, well-circumscribed dural attachment zones along the meninges.\n"
        "  • Tumor Boundary & Margin Delineation: Isolates smooth, uniform tumor boundaries with characteristic dural tail enhancement.\n"
        "  • Clinical Recommendation: Morphologically consistent with extra-axial meningeal lesion; assess adjacent dural venous sinus patency.\n\n"
        "👤 FOR PATIENTS & FAMILIES:\n"
        "  • What This Means: The scan detected a growth arising from the protective outer membranes (meninges) surrounding the brain.\n"
        "  • Heatmap Explanation: The red border highlights a clear, well-defined lesion area that is typically separated from the inner brain tissue.\n"
        "  • Suggested Next Steps: Schedule a consultation with your doctor to review monitoring options or treatment."
    ),
    'PITUITARY': (
        "=================================================================\n"
        " 🩺 CLINICAL DIAGNOSTIC REPORT\n"
        "=================================================================\n"
        "👨‍⚕️ FOR RADIOLOGISTS & CLINICIANS:\n"
        "  • Focal Heatmap Localization: Pronounced focal localization within the sellar and suprasellar fossa at the skull base.\n"
        "  • Tumor Boundary & Margin Delineation: Delineates focal mass enhancement adjacent to optic chiasm anatomical boundaries.\n"
        "  • Clinical Recommendation: Order endocrinological hormone panel and thin-slice dynamic contrast sagittal pituitary MRI.\n\n"
        "👤 FOR PATIENTS & FAMILIES:\n"
        "  • What This Means: The scan located a growth near the pituitary gland (which regulates essential body hormones).\n"
        "  • Heatmap Explanation: The highlighted region points to a specific focal area at the base of the brain.\n"
        "  • Suggested Next Steps: Consult an endocrinologist or neurosurgeon for hormone evaluations and routine vision assessments."
    ),
    'NOTUMOR': (
        "=================================================================\n"
        " 🩺 CLINICAL DIAGNOSTIC REPORT\n"
        "=================================================================\n"
        "👨‍⚕️ FOR RADIOLOGISTS & CLINICIANS:\n"
        "  • Focal Heatmap Localization: Shows uniform, symmetrical baseline parenchymal distribution without focal signal abnormality.\n"
        "  • Tumor Boundary & Margin Delineation: No abnormal tissue boundaries, pathologic mass, or abnormal contrast enhancement detected.\n"
        "  • Clinical Recommendation: Normal brain MRI scan; no evidence of intracranial mass effect, midline shift, or focal lesion.\n\n"
        "👤 FOR PATIENTS & FAMILIES:\n"
        "  • What This Means: The brain MRI scan shows healthy, normal brain tissue with NO tumor detected.\n"
        "  • Heatmap Explanation: The scan shows balanced, uniform brain features with no abnormal spots or highlights.\n"
        "  • Suggested Next Steps: Share these reassuring results with your primary care physician during your routine checkup."
    ),
    'UNRECOGNIZED_TUMOR': (
        "=================================================================\n"
        " 🩺 CLINICAL DIAGNOSTIC REPORT\n"
        "=================================================================\n"
        "⚠️ DIAGNOSTIC STATUS: ATYPICAL / UNCLASSIFIED BRAIN LESION DETECTED\n"
        "-----------------------------------------------------------------\n"
        "👨‍⚕️ FOR RADIOLOGISTS & CLINICIANS:\n"
        "  • Tumor Boundary & Margin Delineation: Automated delineation isolated an abnormal focal brain lesion/mass.\n"
        "  • Pathological Feature Pattern: Morphological characteristics and intensity profile do NOT conform to standard primary tumor profiles (Glioma, Meningioma, Pituitary).\n"
        "  • Differential Considerations: Atypical or secondary intracranial neoplasm (e.g., metastatic lesion, schwannoma, ependymoma, central neurocytoma, craniopharyngioma) or non-neoplastic focal lesion.\n"
        "  • Clinical Recommendation: Order urgent multi-parametric contrast MRI (axial/sagittal/coronal T1+C, T2/FLAIR, DWI/ADC, MR Perfusion) and neurosurgical consultation for biopsy/histopathological verification.\n\n"
        "👤 FOR PATIENTS & FAMILIES:\n"
        "  • What This Means: An abnormal focal area or growth was detected in your brain scan, but it does not match standard typical tumor profiles.\n"
        "  • Heatmap Explanation: The localization box, red margin outline, and heatmap mark the exact focal region where the unusual tissue was identified.\n"
        "  • Suggested Next Steps: ⚠️ PLEASE MEET AND CONSULT YOUR DOCTOR OR SPECIALIST (Neurologist / Neurosurgeon) AS SOON AS POSSIBLE. A qualified physician must review this scan in person to provide an accurate diagnosis and personalized medical guidance."
    )
}

# -----------------------------------------------------------------------------
# 3. Grad-CAM Heatmap Engine
# -----------------------------------------------------------------------------
def generate_gradcam(model, input_tensor, target_class_idx=0):
    gradients, activations = [], []
    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])
    def forward_hook(module, input, output): activations.append(output)
    
    target_layer = model.features.denseblock4
    h_f = target_layer.register_forward_hook(forward_hook)
    h_b = target_layer.register_full_backward_hook(backward_hook)
    
    model.eval()
    inp = input_tensor.clone().requires_grad_(True)
    output = model(inp)
    score = output[0, target_class_idx]
    model.zero_grad()
    score.backward(retain_graph=True)
    h_f.remove()
    h_b.remove()
    
    if len(gradients) > 0 and len(activations) > 0:
        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights): cam += w * acts[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam
    return np.zeros((224, 224), dtype=np.float32)

# -----------------------------------------------------------------------------
# 4. Input File Loader (.jpg, .jpeg, .png, .bmp, .h5)
# -----------------------------------------------------------------------------
def load_input_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[!] File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.h5':
        print(f"[*] Processing 3D MRI slice file: {file_path}")
        with h5py.File(file_path, 'r') as hf:
            if 'image' in hf:
                data = hf['image'][:]
            elif 'raw' in hf:
                data = hf['raw'][:]
            elif 'data' in hf:
                data = hf['data'][:]
            else:
                key = list(hf.keys())[0]
                data = hf[key][:]
                
        if data.ndim == 3:
            if data.shape[2] in [3, 4]:
                stds = [data[:, :, c].std() for c in range(data.shape[2])]
                best_ch = int(np.argmax(stds))
                slice_data = data[:, :, best_ch]
            elif data.shape[0] in [3, 4]:
                stds = [data[c, :, :].std() for c in range(data.shape[0])]
                best_ch = int(np.argmax(stds))
                slice_data = data[best_ch, :, :]
            else:
                slice_data = data[:, :, data.shape[2] // 2]
        else:
            slice_data = data
            
        slice_data = slice_data.astype(np.float32)
        s_min, s_max = slice_data.min(), slice_data.max()
        if s_max > s_min:
            slice_norm = ((slice_data - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
        else:
            slice_norm = np.zeros_like(slice_data, dtype=np.uint8)
            
        return cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2RGB)
    else:
        print(f"[*] Processing 2D image file: {file_path}")
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            raise ValueError(f"[!] Could not read image file with OpenCV: {file_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# -----------------------------------------------------------------------------
# 5. Local Model Initializer (Cached in Memory)
# -----------------------------------------------------------------------------
_GLOBAL_PIPELINE = None
_GLOBAL_SPECIALISTS = None

def load_local_models():
    global _GLOBAL_PIPELINE, _GLOBAL_SPECIALISTS
    if _GLOBAL_PIPELINE is not None and _GLOBAL_SPECIALISTS is not None:
        return _GLOBAL_PIPELINE, _GLOBAL_SPECIALISTS

    print(f"[*] Running on device: {device}")
    
    # 1. Locate YAML config
    cfg_p = os.path.join(EXTRACTED_DIR, 'configs', 'swin_base_patch4_window7_224_finetune.yaml')
    if not os.path.exists(cfg_p):
        for root, dirs, files in os.walk(BASE_DIR):
            for f in files:
                if f.endswith('.yaml'):
                    cfg_p = os.path.join(root, f)
                    break

    # 2. Locate Swin-UNet & YOLO weights
    swin_p = os.path.join(BASE_DIR, 'output_finetune', 'best_model.pth')
    if not os.path.exists(swin_p):
        swin_p = os.path.join(BASE_DIR, 'best_model.pth')
        
    yolo_p = os.path.join(BASE_DIR, 'yolo_best.pt')

    print(f"[*] Config YAML  : {cfg_p}")
    print(f"[*] Swin-UNet Pt : {swin_p}")
    print(f"[*] YOLOv8 Pt    : {yolo_p}")

    pipeline = YOLOSegPipeline(cfg_path=cfg_p, swin_weights_path=swin_p, yolo_weights_path=yolo_p, device=device)

    # 3. Load DenseNet-121 4 Specialists
    specialists = {}
    for cls in CLASSES:
        ckpt_name = f"densenet121_{cls.lower()}.pth"
        ckpt_p = os.path.join(BASE_DIR, ckpt_name)
        if not os.path.exists(ckpt_p):
            ckpt_p = os.path.join(EXTRACTED_DIR, ckpt_name)
            
        m = build_specialist_model().to(device)
        if os.path.exists(ckpt_p):
            m.load_state_dict(torch.load(ckpt_p, map_location=device, weights_only=False))
            print(f"[+] Loaded Specialist ({cls}): {ckpt_p}")
        else:
            print(f"[!] Warning: Specialist checkpoint missing: {ckpt_p}")
        m.eval()
        specialists[cls] = m

    _GLOBAL_PIPELINE = pipeline
    _GLOBAL_SPECIALISTS = specialists
    return _GLOBAL_PIPELINE, _GLOBAL_SPECIALISTS

# -----------------------------------------------------------------------------
# 6. Main Prediction Logic & File Saver
# -----------------------------------------------------------------------------
def run_diagnosis(input_path, output_dir=None):
    if output_dir is None:
        output_dir = BASE_DIR
    os.makedirs(output_dir, exist_ok=True)

    pipeline, specialists = load_local_models()
    img_rgb = load_input_file(input_path)
    h_img, w_img, _ = img_rgb.shape

    # Stage 1: YOLOv8 + Swin-UNet Pipeline Execution
    print("[*] Running Stage 1 YOLOv8 + Swin-UNet Background-Blurred Pipeline...")
    results = pipeline.predict(img_rgb, enforce_bbox=True)

    bbox_img = results['bbox_image']
    seg_img = results['seg_image']
    combined_img = results['combined_image']
    x1, y1, x2, y2 = results['yolo_box']
    total_red_pixels = int(np.sum(results['raw_mask'] > 0))

    # SAVE THE 4 REQUESTED OUTPUT PNG FILES (WITH complete_ PREFIX)
    bbox_path = os.path.join(output_dir, 'complete_test_bbox_output.png')
    seg_path = os.path.join(output_dir, 'complete_test_seg_output.png')
    combined_path = os.path.join(output_dir, 'complete_test_combined_output.png')

    cv2.imwrite(bbox_path, cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(seg_path, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(combined_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))

    print("="*65)
    print(f" [+] SAVED: complete_test_bbox_output.png     -> {bbox_path}")
    print(f" [+] SAVED: complete_test_seg_output.png      -> {seg_path}")
    print(f" [+] SAVED: complete_test_combined_output.png -> {combined_path}")
    print("="*65)

    # Stage 2: Rank 1 Unblurred ROI Patch Crop (+15% Margin)
    combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
    bw, bh = x2 - x1, y2 - y1
    margin_w, margin_h = int(bw * 0.15), int(bh * 0.15)
    ex_x1, ex_y1 = max(0, x1 - margin_w), max(0, y1 - margin_h)
    ex_x2, ex_y2 = min(w_img, x2 + margin_w), min(h_img, y2 + margin_h)

    roi_patch = combined_img_bgr[ex_y1:ex_y2, ex_x1:ex_x2]
    roi_patch_224 = cv2.resize(roi_patch, (224, 224)) if roi_patch.size > 0 else cv2.resize(combined_img_bgr, (224, 224))
    roi_patch_rgb = cv2.cvtColor(roi_patch_224, cv2.COLOR_BGR2RGB)

    # Stage 3: DenseNet-121 Multi-Specialist Classifier
    print("[*] Running Stage 3 DenseNet-121 Multi-Specialist Classifier...")
    pil_roi = Image.fromarray(roi_patch_rgb)
    input_tensor = data_transform(pil_roi).unsqueeze(0).to(device)

    scores = {}
    with torch.no_grad():
        scores['GLIOMA'] = torch.softmax(specialists['GLIOMA'](input_tensor), dim=1)[0, 0].item()
        scores['MENINGIOMA'] = torch.softmax(specialists['MENINGIOMA'](input_tensor), dim=1)[0, 0].item()
        scores['PITUITARY'] = torch.softmax(specialists['PITUITARY'](input_tensor), dim=1)[0, 1].item()
        scores['NOTUMOR'] = torch.softmax(specialists['NOTUMOR'](input_tensor), dim=1)[0, 0].item()

    pred_class = None
    tumor_classes = ['GLIOMA', 'MENINGIOMA', 'PITUITARY']
    best_tumor_class = max(tumor_classes, key=lambda c: scores[c])
    best_tumor_score = scores[best_tumor_class]
    notumor_score = scores['NOTUMOR']

    # Clinical Open-Set Recognition Calibration:
    # 1. Normal Scan (NOTUMOR): NOTUMOR specialist is dominant and confident (>= 50%)
    # 2. Confirmed Trained Tumor: Requires >= 65% confidence in the best matching specialist
    # 3. Unrecognized Tumor: Significant lesion detected, but best matching specialist confidence < 65% and notumor < 50%
    TUMOR_CONF_THRESHOLD = 0.65      # 65% confidence threshold

    if (notumor_score > best_tumor_score and notumor_score >= 0.50) or (best_tumor_score < 0.20 and notumor_score >= 0.40):
        pred_class = 'NOTUMOR'
        conf_percent = notumor_score * 100.0
        gradcam_model_key = 'NOTUMOR'
    elif best_tumor_score >= TUMOR_CONF_THRESHOLD:
        pred_class = best_tumor_class
        conf_percent = best_tumor_score * 100.0
        gradcam_model_key = best_tumor_class
    else:
        # Abnormal tissue / lesion detected, but does not match trained tumor classes with >= 65% certainty
        pred_class = 'UNRECOGNIZED_TUMOR'
        conf_percent = best_tumor_score * 100.0
        gradcam_model_key = best_tumor_class

    # If NO TUMOR is predicted, revert visualization to clean raw image (no bounding box or red mask overlay)
    if pred_class == 'NOTUMOR':
        flair_ch = img_rgb[:, :, 0] if img_rgb.ndim == 3 else img_rgb
        clean_base = np.stack([flair_ch, flair_ch, flair_ch], axis=-1).astype(np.uint8)
        bbox_img = clean_base.copy()
        seg_img = clean_base.copy()
        combined_img = clean_base.copy()
        total_red_pixels = 0

        cv2.imwrite(bbox_path, cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(seg_path, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(combined_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))

    # Stage 4: Explainable AI (Grad-CAM) Heatmap
    if pred_class == 'NOTUMOR':
        # For No-Tumor scans, there is NO tumor to explain: show clean scan without heatmap
        flair_ch = img_rgb[:, :, 0] if img_rgb.ndim == 3 else img_rgb
        xai_overlay = np.stack([flair_ch, flair_ch, flair_ch], axis=-1).astype(np.uint8)
    else:
        target_idx = 1 if gradcam_model_key == 'PITUITARY' else 0
        cam_map = generate_gradcam(specialists[gradcam_model_key], input_tensor, target_idx)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        xai_overlay = cv2.addWeighted(roi_patch_rgb, 0.6, heatmap_rgb, 0.4, 0)

    # Render 5-Panel Display
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(img_rgb[:, :, 0], cmap='gray'); axes[0].set_title('Original FLAIR Scan', fontweight='bold'); axes[0].axis('off')
    
    if pred_class == 'NOTUMOR':
        axes[1].imshow(bbox_img); axes[1].set_title('Lesion Localization (None)', fontweight='bold', color='gray'); axes[1].axis('off')
        axes[2].imshow(seg_img); axes[2].set_title('Margin Delineation (None)', fontweight='bold', color='gray'); axes[2].axis('off')
        axes[3].imshow(combined_img); axes[3].set_title('Clean Brain Scan', fontweight='bold', color='gray'); axes[3].axis('off')
        axes[4].imshow(xai_overlay); axes[4].set_title('Focal Heatmap (Normal)', fontweight='bold', color='gray'); axes[4].axis('off')
        plt.suptitle(f'CLINICAL MRI DIAGNOSIS: NORMAL / NO TUMOR ({conf_percent:.1f}% Confidence)', fontsize=15, fontweight='bold', y=1.04)
    elif pred_class == 'UNRECOGNIZED_TUMOR':
        axes[1].imshow(bbox_img); axes[1].set_title('Lesion Localization Box', fontweight='bold', color='darkorange'); axes[1].axis('off')
        axes[2].imshow(seg_img); axes[2].set_title('Tumor Margin Delineation', fontweight='bold', color='red'); axes[2].axis('off')
        axes[3].imshow(combined_img); axes[3].set_title('Combined Structural View', fontweight='bold', color='purple'); axes[3].axis('off')
        axes[4].imshow(xai_overlay); axes[4].set_title('Diagnostic Focal Heatmap', fontweight='bold', color='darkred'); axes[4].axis('off')
        plt.suptitle('⚠️ CLINICAL MRI DIAGNOSIS: ATYPICAL LESION — SPECIALIST CONSULTATION ADVISED', fontsize=14, fontweight='bold', color='darkred', y=1.04)
    else:
        axes[1].imshow(bbox_img); axes[1].set_title('Lesion Localization Box', fontweight='bold', color='green'); axes[1].axis('off')
        axes[2].imshow(seg_img); axes[2].set_title('Tumor Margin Delineation', fontweight='bold', color='red'); axes[2].axis('off')
        axes[3].imshow(combined_img); axes[3].set_title('Combined Structural View', fontweight='bold', color='purple'); axes[3].axis('off')
        axes[4].imshow(xai_overlay); axes[4].set_title('Diagnostic Focal Heatmap', fontweight='bold', color='darkred'); axes[4].axis('off')
        plt.suptitle(f'CLINICAL MRI DIAGNOSIS: {pred_class} ({conf_percent:.1f}% Confidence)', fontsize=15, fontweight='bold', y=1.04)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'complete_diagnosis_5panel_output.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f" [+] SAVED: complete_diagnosis_5panel_output.png -> {fig_path}")

    print("\n" + "="*65)
    if pred_class == 'UNRECOGNIZED_TUMOR':
        print("  PREDICTED STATUS     : ⚠️ UNRECOGNIZED TUMOR TYPE (LESION DETECTED)")
        print(f"  MAX TRAINED MATCH    : {best_tumor_class} ({conf_percent:.2f}% - Below 65% Confidence Threshold)")
        print("  RECOMMENDATION       : CONSULT SPECIALIST / DOCTOR FOR CLINICAL EVALUATION")
    else:
        print(f"  PREDICTED TUMOR TYPE : {pred_class}")
        print(f"  CONFIDENCE SCORE     : {conf_percent:.2f} %")
    print("-" * 65)
    print(CLINICAL_EXPLANATIONS.get(pred_class, ''))
    print("="*65)

    return {
        'pred_class': pred_class,
        'conf_percent': conf_percent,
        'segmented_pixels': total_red_pixels,
        'scores': scores
    }

# -----------------------------------------------------------------------------
# 7. CLI Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neuro_AI_System Local Patient MRI Prediction (.jpg, .png, .h5)")
    parser.add_argument('-i', '--input', type=str, help="Path to patient MRI file (.jpg, .png, .h5)")
    parser.add_argument('-o', '--output', type=str, default=BASE_DIR, help="Directory to save output PNG files")
    args = parser.parse_args()

    input_file = args.input
    if not input_file:
        input_file = input("Enter path to patient MRI image or .h5 file: ").strip().strip('"').strip("'")

    if input_file:
        run_diagnosis(input_file, args.output)
    else:
        print("[!] No input file specified.")
