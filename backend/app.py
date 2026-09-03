"""
Flask Backend for LungCT AI Capstone Project
Serves the web dashboard and handles REST API requests for upload, prediction, and report download.
Optimized for zero-downtime lazy loading on Cloud Hosts (Render, Railway, Docker).
"""

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_cors import CORS

# Re-configure standard streams for UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Setup directory paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = BACKEND_DIR / "static"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.predict import CTPredictor
from backend.severity import SeverityCalculator
from backend.visualization import CTVisualizer
from backend.report_generator import CTReportGenerator

# --- Model 2 Imports ---
from backend.models.model2.predictor import Model2Predictor
from backend.models.model2.severity import calculate_severity_model2
from backend.models.model2.visualization import Model2Visualizer

# --- Model 3 Imports ---
from backend.models.model3.predictor import Model3Predictor
from backend.models.model3.severity import calculate_severity_model3
from backend.models.model3.visualization import Model3Visualizer

# --- Model 4 Imports ---
from backend.models.model4.predictor import Model4Predictor
from backend.models.model4.severity import calculate_severity_model4
from backend.models.model4.visualization import Model4Visualizer

# Initialize Flask app serving frontend
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="",
    template_folder=str(FRONTEND_DIR)
)

# Security Configuration: 32MB Max Upload Limit
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# Enable Cross-Origin Resource Sharing for all origins (supports Vercel frontend)
CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'dcm'}

def is_safe_image_file(filename: str) -> bool:
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def sanitize_and_save_upload(file):
    """
    Sanitizes, validates, and saves uploaded CT images safely with integrity verification.
    """
    if not file or not file.filename:
        return None, "No file uploaded or filename is empty."

    clean_name = secure_filename(file.filename)
    if not clean_name:
        clean_name = f"ct_scan_{abs(hash(file.filename)) % 100000}.png"

    if not is_safe_image_file(clean_name):
        return None, f"Unsupported file type. Allowed medical image extensions: {', '.join(ALLOWED_EXTENSIONS)}"

    dest_path = UPLOADS_DIR / clean_name
    file.save(str(dest_path))

    # Verify image integrity via OpenCV
    img = cv2.imread(str(dest_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        if dest_path.exists():
            dest_path.unlink()
        return None, "Corrupted or non-decodable medical image payload."

    return clean_name, None

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Payload Too Large. Maximum allowed CT scan size is 32MB."}), 413

MODEL_PATH = BACKEND_DIR / "models" / "model1" / "best_model.pth"
MODEL2_PATH = BACKEND_DIR / "models" / "model2" / "Joshna.pth"
MODEL3_PATH = BACKEND_DIR / "models" / "model3" / "Jahnavi (1).pth"
MODEL4_PATH = BACKEND_DIR / "models" / "model4" / "Vasanth (2).pth"

UPLOADS_DIR = STATIC_DIR / "uploads"
OUTPUTS_DIR = STATIC_DIR / "outputs"
OUTPUTS_M2_DIR = STATIC_DIR / "outputs_m2"
OUTPUTS_M3_DIR = STATIC_DIR / "outputs_m3"
OUTPUTS_M4_DIR = STATIC_DIR / "outputs_m4"
REPORTS_DIR = STATIC_DIR / "reports"

# Ensure all server storage directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_M2_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_M3_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_M4_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy Singleton Model Loaders & Non-blocking Warmup
# ---------------------------------------------------------------------------
_predictor_m1 = None
_predictor_m2 = None
_predictor_m3 = None
_predictor_m4 = None
_model_lock = threading.Lock()

import torch
# Restrict CPU threads to prevent memory spikes on cloud instances
try:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
except Exception:
    pass

def get_predictor_m1():
    global _predictor_m1
    if _predictor_m1 is None:
        with _model_lock:
            if _predictor_m1 is None:
                _predictor_m1 = CTPredictor(str(MODEL_PATH))
                print(f"✅ U-Net++ Model (Model 1) initialized from {MODEL_PATH}")
    return _predictor_m1

def get_predictor_m2():
    global _predictor_m2
    if _predictor_m2 is None:
        with _model_lock:
            if _predictor_m2 is None:
                _predictor_m2 = Model2Predictor(str(MODEL2_PATH))
                print(f"✅ Attention U-Net (Model 2) initialized from {MODEL2_PATH}")
    return _predictor_m2

def get_predictor_m3():
    global _predictor_m3
    if _predictor_m3 is None:
        with _model_lock:
            if _predictor_m3 is None:
                _predictor_m3 = Model3Predictor(str(MODEL3_PATH))
                print(f"✅ DeepLabV3+ (Model 3) initialized from {MODEL3_PATH}")
    return _predictor_m3

def get_predictor_m4():
    global _predictor_m4
    if _predictor_m4 is None:
        with _model_lock:
            if _predictor_m4 is None:
                _predictor_m4 = Model4Predictor(str(MODEL4_PATH))
                print(f"✅ NoiseCNN (Model 4) initialized from {MODEL4_PATH}")
    return _predictor_m4

def is_model1_ready():
    return _predictor_m1 is not None or MODEL_PATH.exists()

def is_model2_ready():
    return _predictor_m2 is not None or MODEL2_PATH.exists()

def is_model3_ready():
    return _predictor_m3 is not None or MODEL3_PATH.exists()

def is_model4_ready():
    return _predictor_m4 is not None or MODEL4_PATH.exists()


import mimetypes

mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/javascript", ".jsx")
mimetypes.add_type("application/javascript", ".ts")
mimetypes.add_type("application/javascript", ".tsx")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/wasm", ".wasm")


@app.route("/")
def index():
    """
    Renders the main clinical dashboard landing page from frontend/ (or built React SPA from frontend/dist).
    """
    dist_index = FRONTEND_DIR / "dist" / "index.html"
    if dist_index.exists():
        return send_file(str(dist_index), mimetype="text/html")
    return send_file(str(FRONTEND_DIR / "index.html"), mimetype="text/html")


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    ext = Path(filename).suffix.lower()
    mime = mimetypes.types_map.get(ext, None)
    if ext in [".js", ".mjs", ".jsx", ".ts", ".tsx"]:
        mime = "application/javascript"
    elif ext == ".css":
        mime = "text/css"
    elif ext == ".svg":
        mime = "image/svg+xml"
    return send_from_directory(str(FRONTEND_DIR / "dist" / "assets"), filename, mimetype=mime)


@app.route("/src/<path:filename>")
def serve_src(filename):
    ext = Path(filename).suffix.lower()
    mime = "application/javascript" if ext in [".js", ".jsx", ".ts", ".tsx"] else "text/plain"
    return send_from_directory(str(FRONTEND_DIR / "src"), filename, mimetype=mime)


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(str(FRONTEND_DIR / "css"), filename, mimetype="text/css")


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(str(FRONTEND_DIR / "js"), filename, mimetype="application/javascript")


@app.route("/static/<path:filename>")
def serve_static(filename):
    ext = Path(filename).suffix.lower()
    mime = mimetypes.types_map.get(ext, None)
    return send_from_directory(str(STATIC_DIR), filename, mimetype=mime)


@app.route("/health", methods=["GET"])
def health():
    """
    API Health check endpoint returning status of all 4 deployed models.
    """
    return jsonify({
        "status": "healthy",
        "models": {
            "model1": {
                "loaded": is_model1_ready(),
                "name": "U-Net++",
                "path": str(MODEL_PATH),
                "classes": ["Gaussian", "Poisson"],
            },
            "model2": {
                "loaded": is_model2_ready(),
                "name": "Attention U-Net",
                "path": str(MODEL2_PATH),
                "classes": ["Poisson", "Speckle"],
            },
            "model3": {
                "loaded": is_model3_ready(),
                "name": "DeepLabV3+",
                "path": str(MODEL3_PATH),
                "classes": ["Salt & Pepper", "RVIN"],
            },
            "model4": {
                "loaded": is_model4_ready(),
                "name": "NoiseCNN",
                "path": str(MODEL4_PATH),
                "classes": ["Quantization", "Periodic"],
            },
        },
        "model_loaded": is_model1_ready(),
        "model2_loaded": is_model2_ready(),
        "model3_loaded": is_model3_ready(),
        "model4_loaded": is_model4_ready(),
    })


@app.route("/api", methods=["GET"])
@app.route("/api/v1", methods=["GET"])
def api_docs():
    """
    Returns complete OpenAPI / REST API documentation for all deployed endpoints.
    """
    return jsonify({
        "name": "LungCT AI REST API",
        "version": "2.0.0",
        "description": "Multi-Model CT Noise Classification, Severity Quantification & PDF Reporting Engine",
        "endpoints": {
            "GET /health": {
                "description": "System health check and status of all 4 deep learning models",
                "returns": "JSON object with status, model loading states, and class definitions"
            },
            "POST /upload": {
                "description": "Upload a raw CT scan slice image",
                "content_type": "multipart/form-data",
                "body": {"file": "Binary image file (PNG/JPEG/BMP)"},
                "returns": "JSON with uploaded filename and static preview URL"
            },
            "POST /predict": {
                "alias": "POST /predict/model1",
                "description": "Execute Model 1 (U-Net++) inference for Gaussian and Poisson noise segmentation",
                "content_type": "multipart/form-data",
                "body": {"file": "Binary image file"},
                "returns": "JSON with pixel counts, percentage coverage, severity levels, bounding boxes, and image URLs"
            },
            "POST /predict/model2": {
                "description": "Execute Model 2 (Attention U-Net) inference for Poisson and Speckle noise segmentation",
                "content_type": "multipart/form-data",
                "body": {"file": "Binary image file"},
                "returns": "JSON with Attention-gated masks, overlays, and severity breakdown"
            },
            "POST /predict/model3": {
                "description": "Execute Model 3 (DeepLabV3+ ASPP) inference for Salt & Pepper and RVIN noise segmentation",
                "content_type": "multipart/form-data",
                "body": {"file": "Binary image file"},
                "returns": "JSON with ASPP multi-scale impulse masks, overlays, and severity metrics"
            },
            "POST /predict/model4": {
                "description": "Execute Model 4 (NoiseCNN) inference for Quantization and Periodic noise classification",
                "content_type": "multipart/form-data",
                "body": {"file": "Binary image file"},
                "returns": "JSON with softmax probabilities, confidence score, and 2D FFT Fourier spectrum image URL"
            },
            "GET /demo/model1": {
                "alias": "GET /demo",
                "description": "Synthesize a realistic CT scan phantom with Gaussian/Poisson noise and run Model 1"
            },
            "GET /demo/model2": {
                "description": "Synthesize a CT phantom with Poisson/Speckle noise and run Model 2 (Attention U-Net)"
            },
            "GET /demo/model3": {
                "description": "Synthesize a CT phantom with Salt & Pepper/RVIN noise and run Model 3 (DeepLabV3+)"
            },
            "GET /demo/model4": {
                "description": "Synthesize a CT phantom with Quantization/Periodic noise and run Model 4 (NoiseCNN)"
            },
            "GET /report": {
                "description": "Generate and download a clinical ReportLab PDF report for any diagnosed scan",
                "query_params": {
                    "filename": "Required. Original filename (e.g. scan.png)",
                    "model": "Optional. 'model1', 'model2', 'model3', or 'model4' (default: 'model1')"
                },
                "returns": "Downloadable application/pdf file stream"
            },
            "POST /batch": {
                "description": "Batch inference pipeline for multiple CT images",
                "content_type": "multipart/form-data",
                "body": {"files": "Array of image files"},
                "returns": "JSON list of inference results per image"
            }
        }
    })


@app.route("/demo", methods=["GET"])
@app.route("/demo/model1", methods=["GET"])
def demo():
    """
    Generates a realistic CT scan phantom with noise, runs prediction, and returns analysis.
    """
    try:
        from scripts.generate_realistic_ct import create_realistic_ct_phantom
        
        # Generate demo file
        filename = "demo_phantom.png"
        file_path = UPLOADS_DIR / filename
        create_realistic_ct_phantom(str(file_path))
        
        # Read image using OpenCV (grayscale)
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Failed to read generated demo scan."}), 500
            
        # Step 1: Model Segmentation
        predictor_m1 = get_predictor_m1()
        predicted_mask = predictor_m1.predict(str(file_path))
        
        # Step 2: Severity Assessment
        calculator = SeverityCalculator()
        severity_report = calculator.get_detailed_report(predicted_mask)
        
        # Step 3: Draw Diagnostic Bounding Boxes and Region Highlights
        visualizer = CTVisualizer(image)
        output_filename = f"{Path(filename).stem}_result.png"
        output_path = OUTPUTS_DIR / output_filename
        
        visuals = visualizer.generate_full_visualization(
            predicted_mask,
            severity_report,
            str(output_path)
        )
        
        overlay_filename = f"{Path(filename).stem}_overlay.png"
        
        response_data = {
            "success": True,
            "filename": filename,
            "gaussian": severity_report["gaussian"]["percentage"],
            "poisson": severity_report["poisson"]["percentage"],
            "gaussian_level": severity_report["gaussian"]["level"],
            "poisson_level": severity_report["poisson"]["level"],
            "total_noise": severity_report["summary"]["total_noise_percentage"],
            "total_level": severity_report["summary"]["total_noise_level"],
            "original_url": f"/static/uploads/{filename}",
            "annotated_url": f"/static/outputs/{output_filename}",
            "overlay_url": f"/static/outputs/{overlay_filename}",
            "pixels": severity_report["pixels"],
            "regions": visuals.get("regions", [])
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Demo Pipeline Error: {e}")
        return jsonify({"error": f"Failed to run demo pipeline: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """
    Endpoint for uploading a raw Lung CT scan image with security validation.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file key found in multipart/form-data"}), 400
        
    file = request.files["file"]
    clean_name, err = sanitize_and_save_upload(file)
    if err:
        return jsonify({"error": err}), 400
        
    return jsonify({
        "success": True,
        "filename": clean_name,
        "url": f"/static/uploads/{clean_name}"
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Run Model 1 (U-Net++) machine learning inference pipeline on upload.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file key found in multipart/form-data"}), 400
        
    file = request.files["file"]
    clean_name, err = sanitize_and_save_upload(file)
    if err:
        return jsonify({"error": err}), 400
        
    file_path = UPLOADS_DIR / clean_name
    try:
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Invalid file format or corrupted medical image."}), 400
            
        predictor_m1 = get_predictor_m1()
        predicted_mask = predictor_m1.predict(str(file_path))
        
        calculator = SeverityCalculator()
        severity_report = calculator.get_detailed_report(predicted_mask)
        
        visualizer = CTVisualizer(image)
        output_filename = f"{Path(clean_name).stem}_result.png"
        output_path = OUTPUTS_DIR / output_filename
        
        visuals = visualizer.generate_full_visualization(
            predicted_mask,
            severity_report,
            str(output_path)
        )
        
        overlay_filename = f"{Path(clean_name).stem}_overlay.png"
        
        return jsonify({
            "success": True,
            "filename": clean_name,
            "gaussian": severity_report["gaussian"]["percentage"],
            "poisson": severity_report["poisson"]["percentage"],
            "gaussian_level": severity_report["gaussian"]["level"],
            "poisson_level": severity_report["poisson"]["level"],
            "total_noise": severity_report["summary"]["total_noise_percentage"],
            "total_level": severity_report["summary"]["total_noise_level"],
            "original_url": f"/static/uploads/{clean_name}",
            "annotated_url": f"/static/outputs/{output_filename}",
            "overlay_url": f"/static/outputs/{overlay_filename}",
            "pixels": severity_report["pixels"],
            "regions": visuals.get("regions", [])
        })
        
    except Exception as e:
        print(f"Prediction Pipeline Error: {e}")
        return jsonify({"error": f"AI inference failed: {str(e)}"}), 500


@app.route("/predict/model2", methods=["POST"])
def predict_model2():
    """
    Standalone Model 2 (Attention U-Net) inference endpoint.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file key found in multipart/form-data"}), 400

    file = request.files["file"]
    clean_name, err = sanitize_and_save_upload(file)
    if err:
        return jsonify({"error": err}), 400

    file_path = UPLOADS_DIR / clean_name
    try:
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Invalid file format or corrupted image."}), 400

        predictor_m2 = get_predictor_m2()
        m2_mask = predictor_m2.predict(str(file_path))
        m2_report = calculate_severity_model2(m2_mask)
        m2_visualizer = Model2Visualizer(image)
        stem = Path(clean_name).stem
        m2_output_filename = f"{stem}_m2_result.png"
        m2_output_path = OUTPUTS_M2_DIR / m2_output_filename
        m2_visualizer.generate_full_visualization(m2_mask, m2_report, str(m2_output_path))

        return jsonify({
            "success": True,
            "model": "Model 2",
            "architecture": "Attention U-Net",
            "filename": clean_name,
            "original_url": f"/static/uploads/{clean_name}",
            "noise": m2_report["noise"],
            "summary": m2_report["summary"],
            "images": {
                "annotated": f"/static/outputs_m2/{m2_output_filename}",
                "overlay":   f"/static/outputs_m2/{stem}_m2_result_overlay.png",
                "mask":      f"/static/outputs_m2/{stem}_m2_result_mask.png",
            }
        })

    except Exception as e:
        print(f"Model 2 Standalone Predict Error: {e}")
        return jsonify({"error": f"Model 2 inference failed: {str(e)}"}), 500


@app.route("/demo/model2", methods=["GET"])
def demo_model2():
    """
    Standalone Model 2 demo — generates a CT phantom and runs Attention U-Net.
    """
    try:
        from scripts.generate_realistic_ct import create_realistic_ct_phantom_model2

        filename = "demo_phantom.png"
        file_path = UPLOADS_DIR / filename
        create_realistic_ct_phantom_model2(str(file_path))

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Failed to read generated demo scan."}), 500

        predictor_m2 = get_predictor_m2()
        m2_mask = predictor_m2.predict(str(file_path))
        m2_report = calculate_severity_model2(m2_mask)
        m2_visualizer = Model2Visualizer(image)
        stem = Path(filename).stem
        m2_output_filename = f"{stem}_m2_result.png"
        m2_output_path = OUTPUTS_M2_DIR / m2_output_filename
        m2_visualizer.generate_full_visualization(m2_mask, m2_report, str(m2_output_path))

        return jsonify({
            "success": True,
            "model": "Model 2",
            "architecture": "Attention U-Net",
            "filename": filename,
            "original_url": f"/static/uploads/{filename}",
            "noise": m2_report["noise"],
            "summary": m2_report["summary"],
            "images": {
                "annotated": f"/static/outputs_m2/{m2_output_filename}",
                "overlay":   f"/static/outputs_m2/{stem}_m2_result_overlay.png",
                "mask":      f"/static/outputs_m2/{stem}_m2_result_mask.png",
            }
        })

    except Exception as e:
        print(f"Model 2 Demo Error: {e}")
        return jsonify({"error": f"Model 2 demo failed: {str(e)}"}), 500


@app.route("/predict/model3", methods=["POST"])
def predict_model3():
    """
    Standalone Model 3 (DeepLabV3+) inference endpoint.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file key found in multipart/form-data"}), 400

    file = request.files["file"]
    clean_name, err = sanitize_and_save_upload(file)
    if err:
        return jsonify({"error": err}), 400

    file_path = UPLOADS_DIR / clean_name
    try:
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Invalid file format or corrupted image."}), 400

        predictor_m3 = get_predictor_m3()
        m3_mask = predictor_m3.predict(str(file_path))
        m3_report = calculate_severity_model3(m3_mask)
        m3_visualizer = Model3Visualizer(image)
        stem = Path(clean_name).stem
        m3_output_filename = f"{stem}_m3_result.png"
        m3_output_path = OUTPUTS_M3_DIR / m3_output_filename
        m3_visualizer.generate_full_visualization(m3_mask, m3_report, str(m3_output_path))

        return jsonify({
            "success": True,
            "model": "Model 3",
            "architecture": "DeepLabV3+",
            "filename": clean_name,
            "original_url": f"/static/uploads/{clean_name}",
            "noise": m3_report["noise"],
            "summary": m3_report["summary"],
            "images": {
                "annotated": f"/static/outputs_m3/{m3_output_filename}",
                "overlay":   f"/static/outputs_m3/{stem}_m3_result_overlay.png",
                "mask":      f"/static/outputs_m3/{stem}_m3_result_mask.png",
            }
        })

    except Exception as e:
        print(f"Model 3 Standalone Predict Error: {e}")
        return jsonify({"error": f"Model 3 inference failed: {str(e)}"}), 500


@app.route("/demo/model3", methods=["GET"])
def demo_model3():
    """
    Standalone Model 3 demo — generates a CT phantom and runs DeepLabV3+.
    """
    try:
        from scripts.generate_realistic_ct import create_realistic_ct_phantom

        filename = "demo_phantom.png"
        file_path = UPLOADS_DIR / filename
        create_realistic_ct_phantom(str(file_path))

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Failed to read generated demo scan."}), 500

        predictor_m3 = get_predictor_m3()
        m3_mask = predictor_m3.predict(str(file_path))
        m3_report = calculate_severity_model3(m3_mask)
        m3_visualizer = Model3Visualizer(image)
        stem = Path(filename).stem
        m3_output_filename = f"{stem}_m3_result.png"
        m3_output_path = OUTPUTS_M3_DIR / m3_output_filename
        m3_visualizer.generate_full_visualization(m3_mask, m3_report, str(m3_output_path))

        return jsonify({
            "success": True,
            "model": "Model 3",
            "architecture": "DeepLabV3+",
            "filename": filename,
            "original_url": f"/static/uploads/{filename}",
            "noise": m3_report["noise"],
            "summary": m3_report["summary"],
            "images": {
                "annotated": f"/static/outputs_m3/{m3_output_filename}",
                "overlay":   f"/static/outputs_m3/{stem}_m3_result_overlay.png",
                "mask":      f"/static/outputs_m3/{stem}_m3_result_mask.png",
            }
        })

    except Exception as e:
        print(f"Model 3 Demo Error: {e}")
        return jsonify({"error": f"Model 3 demo failed: {str(e)}"}), 500


@app.route("/predict/model4", methods=["POST"])
def predict_model4():
    """
    Standalone Model 4 (NoiseCNN) inference endpoint.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file key found in multipart/form-data"}), 400

    file = request.files["file"]
    clean_name, err = sanitize_and_save_upload(file)
    if err:
        return jsonify({"error": err}), 400

    file_path = UPLOADS_DIR / clean_name
    try:
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Invalid file format or corrupted image."}), 400

        predictor_m4 = get_predictor_m4()
        m4_probs = predictor_m4.predict(str(file_path))
        m4_report = calculate_severity_model4(m4_probs)
        m4_visualizer = Model4Visualizer(image)
        stem = Path(clean_name).stem
        m4_output_filename = f"{stem}_m4_result.png"
        m4_output_path = OUTPUTS_M4_DIR / m4_output_filename
        m4_visualizer.generate_full_visualization(m4_probs, m4_report, str(m4_output_path))

        return jsonify({
            "success": True,
            "model": "Model 4",
            "architecture": "NoiseCNN",
            "filename": clean_name,
            "original_url": f"/static/uploads/{clean_name}",
            "predicted_class": m4_report["predicted_class"],
            "confidence": m4_report["confidence"],
            "noise": m4_report["noise"],
            "summary": m4_report["summary"],
            "images": {
                "annotated": f"/static/outputs_m4/{m4_output_filename}",
                "overlay":   f"/static/outputs_m4/{stem}_m4_result_overlay.png",
                "spectrum":  f"/static/outputs_m4/{stem}_m4_result_spectrum.png",
            }
        })

    except Exception as e:
        print(f"Model 4 Standalone Predict Error: {e}")
        return jsonify({"error": f"Model 4 inference failed: {str(e)}"}), 500


@app.route("/demo/model4", methods=["GET"])
def demo_model4():
    """
    Standalone Model 4 demo — generates a CT phantom and runs NoiseCNN.
    """
    try:
        from scripts.generate_realistic_ct import create_realistic_ct_phantom

        filename = "demo_phantom.png"
        file_path = UPLOADS_DIR / filename
        create_realistic_ct_phantom(str(file_path))

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Failed to read generated demo scan."}), 500

        predictor_m4 = get_predictor_m4()
        m4_probs = predictor_m4.predict(str(file_path))
        m4_report = calculate_severity_model4(m4_probs)
        m4_visualizer = Model4Visualizer(image)
        stem = Path(filename).stem
        m4_output_filename = f"{stem}_m4_result.png"
        m4_output_path = OUTPUTS_M4_DIR / m4_output_filename
        m4_visualizer.generate_full_visualization(m4_probs, m4_report, str(m4_output_path))

        return jsonify({
            "success": True,
            "model": "Model 4",
            "architecture": "NoiseCNN",
            "filename": filename,
            "original_url": f"/static/uploads/{filename}",
            "predicted_class": m4_report["predicted_class"],
            "confidence": m4_report["confidence"],
            "noise": m4_report["noise"],
            "summary": m4_report["summary"],
            "images": {
                "annotated": f"/static/outputs_m4/{m4_output_filename}",
                "overlay":   f"/static/outputs_m4/{stem}_m4_result_overlay.png",
                "spectrum":  f"/static/outputs_m4/{stem}_m4_result_spectrum.png",
            }
        })

    except Exception as e:
        print(f"Model 4 Demo Error: {e}")
        return jsonify({"error": f"Model 4 demo failed: {str(e)}"}), 500


@app.route("/report", methods=["GET"])
def report():
    """
    Generate downloadable PDF clinical report for a diagnosed image.
    Supports Model 1, Model 2, Model 3, and Model 4.
    """
    filename = request.args.get("filename")
    model_key = request.args.get("model", "model1").lower()

    if not filename:
        return jsonify({"error": "Missing required query parameter: filename"}), 400
        
    original_path = UPLOADS_DIR / filename
    if not original_path.exists():
        return jsonify({"error": "Requested original CT scan was not found on server."}), 404
        
    stem = Path(filename).stem
    pdf_filename = f"{stem}_{model_key}_report.pdf"
    pdf_path = REPORTS_DIR / pdf_filename
    
    try:
        if model_key in ["model2", "m2"]:
            predictor_m2 = get_predictor_m2()
            annotated_path = OUTPUTS_M2_DIR / f"{stem}_m2_result.png"
            if not annotated_path.exists():
                annotated_path = OUTPUTS_M2_DIR / f"{stem}_result.png"
            m2_mask = predictor_m2.predict(str(original_path))
            severity_report = calculate_severity_model2(m2_mask)
            
            if not annotated_path.exists():
                img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
                Model2Visualizer(img).generate_full_visualization(m2_mask, severity_report, str(annotated_path))

        elif model_key in ["model3", "m3"]:
            predictor_m3 = get_predictor_m3()
            annotated_path = OUTPUTS_M3_DIR / f"{stem}_m3_result.png"
            if not annotated_path.exists():
                annotated_path = OUTPUTS_M3_DIR / f"{stem}_result.png"
            m3_mask = predictor_m3.predict(str(original_path))
            severity_report = calculate_severity_model3(m3_mask)
            
            if not annotated_path.exists():
                img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
                Model3Visualizer(img).generate_full_visualization(m3_mask, severity_report, str(annotated_path))

        elif model_key in ["model4", "m4"]:
            predictor_m4 = get_predictor_m4()
            annotated_path = OUTPUTS_M4_DIR / f"{stem}_m4_result.png"
            if not annotated_path.exists():
                annotated_path = OUTPUTS_M4_DIR / f"{stem}_result.png"
            m4_probs = predictor_m4.predict(str(original_path))
            severity_report = calculate_severity_model4(m4_probs)
            
            if not annotated_path.exists():
                img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
                Model4Visualizer(img).generate_full_visualization(m4_probs, severity_report, str(annotated_path))

        else: # Model 1 (U-Net++)
            predictor_m1 = get_predictor_m1()
            annotated_path = OUTPUTS_DIR / f"{stem}_result.png"
            predicted_mask = predictor_m1.predict(str(original_path))
            calculator = SeverityCalculator()
            severity_report = calculator.get_detailed_report(predicted_mask)
            
            if not annotated_path.exists():
                img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
                CTVisualizer(img).generate_full_visualization(predicted_mask, severity_report, str(annotated_path))

        # Build ReportLab PDF
        CTReportGenerator.generate_pdf(
            output_pdf_path=str(pdf_path),
            original_img_path=str(original_path),
            annotated_img_path=str(annotated_path),
            report_data=severity_report,
            filename=filename,
            model_name=model_key
        )
        
        return send_file(str(pdf_path), as_attachment=True, download_name=pdf_filename)
        
    except Exception as e:
        print(f"Report Generation Error ({model_key}): {e}")
        return jsonify({"error": f"Failed to generate clinical PDF: {str(e)}"}), 500


@app.route("/batch", methods=["POST"])
def batch_predict():
    """
    Batch inference pipeline for multiple CT images.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files key found in multipart/form-data"}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "No files uploaded"}), 400

    results = []
    predictor_m1 = get_predictor_m1()
    calculator = SeverityCalculator()

    for file in files:
        clean_name, err = sanitize_and_save_upload(file)
        if err:
            results.append({"filename": file.filename, "error": err})
            continue

        file_path = UPLOADS_DIR / clean_name
        try:
            image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                results.append({"filename": clean_name, "error": "Invalid image format"})
                continue

            predicted_mask = predictor_m1.predict(str(file_path))
            severity_report = calculator.get_detailed_report(predicted_mask)
            visualizer = CTVisualizer(image)
            output_filename = f"{Path(clean_name).stem}_result.png"
            output_path = OUTPUTS_DIR / output_filename
            visualizer.generate_full_visualization(predicted_mask, severity_report, str(output_path))

            results.append({
                "filename": clean_name,
                "gaussian": severity_report["gaussian"]["percentage"],
                "poisson": severity_report["poisson"]["percentage"],
                "total_noise": severity_report["summary"]["total_noise_percentage"],
                "total_level": severity_report["summary"]["total_noise_level"],
                "annotated_url": f"/static/outputs/{output_filename}"
            })
        except Exception as e:
            results.append({"filename": clean_name, "error": str(e)})

    return jsonify({"success": True, "count": len(results), "results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Launching LungCT AI Flask server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
