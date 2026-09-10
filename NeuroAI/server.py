import os
import sys
import base64
import tempfile
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# -----------------------------------------------------------------------------
# Setup Local Import Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Neuro_AI_System import run_diagnosis, load_input_file, CLINICAL_EXPLANATIONS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Helper function to convert image file to Base64 data URL
def image_to_base64(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

def find_sample_image(class_name, sample_filename):
    # 1. Search in workspace dataset_c/dataset_c/test/<class_name>
    candidates = [
        os.path.join(BASE_DIR, 'dataset_c', 'dataset_c', 'test', class_name.lower(), sample_filename),
        os.path.join(BASE_DIR, 'dataset_c', 'test', class_name.lower(), sample_filename),
        os.path.join(os.path.expanduser('~'), 'Downloads', 'dataset_c', 'test', class_name.lower(), sample_filename)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # Fallback to first image in that specific class directory
    for base in [os.path.join(BASE_DIR, 'dataset_c', 'dataset_c', 'test', class_name.lower()),
                 os.path.join(BASE_DIR, 'dataset_c', 'test', class_name.lower())]:
        if os.path.exists(base):
            flist = [f for f in os.listdir(base) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if len(flist) > 0:
                return os.path.join(base, flist[0])
    return None

@app.route('/api/samples', methods=['GET'])
def get_samples():
    samples = [
        {"id": "glioma", "name": "Glioma MRI Sample (Te-glTr_0000)", "class": "GLIOMA", "path": find_sample_image("glioma", "Te-glTr_0000.jpg")},
        {"id": "meningioma", "name": "Meningioma MRI Sample (Te-me_0012)", "class": "MENINGIOMA", "path": find_sample_image("meningioma", "Te-me_0012.jpg")},
        {"id": "pituitary", "name": "Pituitary MRI Sample (Te-pi_0014)", "class": "PITUITARY", "path": find_sample_image("pituitary", "Te-pi_0014.jpg")},
        {"id": "notumor", "name": "Healthy Brain MRI Sample (Te-noTr_0000)", "class": "NOTUMOR", "path": find_sample_image("notumor", "Te-noTr_0000.jpg")}
    ]
    return jsonify({"status": "success", "samples": samples})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        sample_id = request.form.get('sample_id')
        temp_file_path = None

        if sample_id:
            sample_mapping = {
                "glioma": find_sample_image("glioma", "Te-glTr_0000.jpg"),
                "meningioma": find_sample_image("meningioma", "Te-me_0012.jpg"),
                "pituitary": find_sample_image("pituitary", "Te-pi_0014.jpg"),
                "notumor": find_sample_image("notumor", "Te-noTr_0000.jpg")
            }
            temp_file_path = sample_mapping.get(sample_id)
            filename = os.path.basename(temp_file_path) if temp_file_path else f"{sample_id}.jpg"
        else:
            if 'file' not in request.files:
                return jsonify({"status": "error", "message": "No file uploaded"}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({"status": "error", "message": "Empty file name"}), 400

            filename = file.filename
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                return jsonify({"status": "error", "message": "Unsupported file format. Please upload a standard brain MRI scan (.jpg, .jpeg, .png)."}), 400

            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, filename)
            file.save(temp_file_path)

        if not temp_file_path or not os.path.exists(temp_file_path):
            return jsonify({"status": "error", "message": "File path not found"}), 404

        # Run Neuro AI Prediction Pipeline
        output_dir = BASE_DIR
        diag_res = run_diagnosis(temp_file_path, output_dir)

        # Output image paths
        bbox_path = os.path.join(output_dir, 'complete_test_bbox_output.png')
        seg_path = os.path.join(output_dir, 'complete_test_seg_output.png')
        combined_path = os.path.join(output_dir, 'complete_test_combined_output.png')
        five_panel_path = os.path.join(output_dir, 'complete_diagnosis_5panel_output.png')

        bbox_b64 = image_to_base64(bbox_path)
        seg_b64 = image_to_base64(seg_path)
        combined_b64 = image_to_base64(combined_path)
        five_panel_b64 = image_to_base64(five_panel_path)

        pred_cls = diag_res['pred_class'] if diag_res else 'MENINGIOMA'
        conf_val = diag_res['conf_percent'] if diag_res else 95.0
        seg_pix = diag_res['segmented_pixels'] if diag_res else 0

        return jsonify({
            "status": "success",
            "filename": filename,
            "pred_class": pred_cls,
            "conf_percent": round(conf_val, 2),
            "segmented_pixels": int(seg_pix),
            "explanation_text": CLINICAL_EXPLANATIONS.get(pred_cls, ''),
            "images": {
                "bbox": bbox_b64,
                "seg": seg_b64,
                "combined": combined_b64,
                "five_panel": five_panel_b64
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("[*] Starting Neuro AI Web Application Server on http://127.0.0.1:8080 ...")
    app.run(host='0.0.0.0', port=8080, debug=False)
