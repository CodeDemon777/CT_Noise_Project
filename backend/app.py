"""
Flask Backend for LungCT AI Capstone Project
Serves the web dashboard and handles REST API requests for upload, prediction, and report download.
"""

import os
import sys
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

# Enable Cross-Origin Resource Sharing
CORS(app)

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
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
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

# Load U-Net++ predictor on startup (Model 1)
try:
    predictor = CTPredictor(str(MODEL_PATH))
    model_loaded = True
    print(f"✅ U-Net++ Model (Model 1) loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load Model 1: {e}")
    model_loaded = False

# Load Attention U-Net predictor on startup (Model 2)
try:
    predictor_m2 = Model2Predictor(str(MODEL2_PATH))
    model2_loaded = True
    print(f"✅ Attention U-Net (Model 2) loaded successfully from {MODEL2_PATH}")
except Exception as e:
    print(f"❌ Failed to load Model 2: {e}")
    predictor_m2 = None
    model2_loaded = False

# Load DeepLabV3+ predictor on startup (Model 3)
try:
    predictor_m3 = Model3Predictor(str(MODEL3_PATH))
    model3_loaded = True
    print(f"✅ DeepLabV3+ (Model 3) loaded successfully from {MODEL3_PATH}")
except Exception as e:
    print(f"❌ Failed to load Model 3: {e}")
    predictor_m3 = None
    model3_loaded = False

# Load NoiseCNN predictor on startup (Model 4)
try:
    predictor_m4 = Model4Predictor(str(MODEL4_PATH))
    model4_loaded = True
    print(f"✅ NoiseCNN (Model 4) loaded successfully from {MODEL4_PATH}")
except Exception as e:
    print(f"❌ Failed to load Model 4: {e}")
    predictor_m4 = None
    model4_loaded = False


@app.route("/")
def index():
    """
    Renders the main clinical dashboard landing page from frontend/ (or built React SPA from frontend/dist).
    """
    dist_index = FRONTEND_DIR / "dist" / "index.html"
    if dist_index.exists():
        return send_file(str(dist_index))
    return send_file(str(FRONTEND_DIR / "index.html"))


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(str(FRONTEND_DIR / "dist" / "assets"), filename)


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(str(FRONTEND_DIR / "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(str(FRONTEND_DIR / "js"), filename)


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(str(STATIC_DIR), filename)


@app.route("/health", methods=["GET"])
def health():
    """
    API Health check endpoint returning status of all 4 deployed models.
    """
    return jsonify({
        "status": "healthy",
        "models": {
            "model1": {
                "loaded": model_loaded,
                "name": "U-Net++",
                "path": str(MODEL_PATH),
                "classes": ["Gaussian", "Poisson"],
            },
            "model2": {
                "loaded": model2_loaded,
                "name": "Attention U-Net",
                "path": str(MODEL2_PATH),
                "classes": ["Poisson", "Speckle"],
            },
            "model3": {
                "loaded": model3_loaded,
                "name": "DeepLabV3+",
                "path": str(MODEL3_PATH),
                "classes": ["Salt & Pepper", "RVIN"],
            },
            "model4": {
                "loaded": model4_loaded,
                "name": "NoiseCNN",
                "path": str(MODEL4_PATH),
                "classes": ["Quantization", "Periodic"],
            },
        },
        "model_loaded": model_loaded,
        "model2_loaded": model2_loaded,
        "model3_loaded": model3_loaded,
        "model4_loaded": model4_loaded,
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
    Allows testing the app immediately without uploading an external file.
    """
    if not model_loaded:
        return jsonify({"error": "Inference server state error: U-Net++ model is not loaded."}), 500
        
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
        predicted_mask = predictor.predict(str(file_path))
        
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
        
        # Generate overlay image URL
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
        
        # --- MODEL 2 (Attention U-Net) Independent Pipeline ---
        m2_data = None
        if model2_loaded and predictor_m2 is not None:
            try:
                m2_mask = predictor_m2.predict(str(file_path))
                m2_report = calculate_severity_model2(m2_mask)
                m2_visualizer = Model2Visualizer(image)
                m2_output_filename = f"{Path(filename).stem}_m2_result.png"
                m2_output_path = OUTPUTS_M2_DIR / m2_output_filename
                m2_visuals = m2_visualizer.generate_full_visualization(
                    m2_mask, m2_report, str(m2_output_path)
                )
                stem = Path(filename).stem
                m2_data = {
                    "model": "Model 2",
                    "architecture": "Attention U-Net",
                    "noise": m2_report["noise"],
                    "summary": m2_report["summary"],
                    "images": {
                        "mask": f"/static/outputs_m2/{stem}_m2_result_mask.png",
                        "overlay": f"/static/outputs_m2/{stem}_m2_result_overlay.png",
                        "annotated": f"/static/outputs_m2/{m2_output_filename}",
                    }
                }
            except Exception as e2:
                print(f"Model 2 Demo Pipeline Error: {e2}")
                m2_data = {"error": str(e2)}

        response_data["model2"] = m2_data
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Demo Pipeline Error: {e}")
        return jsonify({"error": f"Failed to run demo pipeline: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    """
    Endpoint for uploading a raw Lung CT scan image.
    Saves image under /static/uploads/ for diagnostic preview.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file key found in multipart/form-data"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Selected filename is empty"}), 400
        
    try:
        file_path = UPLOADS_DIR / file.filename
        file.save(str(file_path))
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "url": f"/static/uploads/{file.filename}"
        })
    except Exception as e:
        return jsonify({"error": f"Upload write failed: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Run machine learning inference pipeline on upload.
    Pipeline: Input Upload -> Preprocessing -> Model Predict -> Severity Calculator -> Visualization Overlay -> JSON response.
    """
    if not model_loaded:
        return jsonify({"error": "Inference server state error: U-Net++ model is not loaded."}), 500
        
    if "file" not in request.files:
        return jsonify({"error": "No file key found in multipart/form-data"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Selected filename is empty"}), 400
        
    try:
        # Save file to uploads folder
        file_path = UPLOADS_DIR / file.filename
        file.save(str(file_path))
        
        # Read image using OpenCV (grayscale for medical imaging profiles)
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Invalid file format or corrupted medical image."}), 400
            
        # Step 1: Model Segmentation
        predicted_mask = predictor.predict(str(file_path))
        
        # Step 2: Severity Assessment
        calculator = SeverityCalculator()
        severity_report = calculator.get_detailed_report(predicted_mask)
        
        # Step 3: Draw Diagnostic Bounding Boxes and Region Highlights
        visualizer = CTVisualizer(image)
        output_filename = f"{Path(file.filename).stem}_result.png"
        output_path = OUTPUTS_DIR / output_filename
        
        visuals = visualizer.generate_full_visualization(
            predicted_mask,
            severity_report,
            str(output_path)
        )
        
        # Generate overlay image URL
        overlay_filename = f"{Path(file.filename).stem}_overlay.png"
        
        response_data = {
            "success": True,
            "filename": file.filename,
            "gaussian": severity_report["gaussian"]["percentage"],
            "poisson": severity_report["poisson"]["percentage"],
            "gaussian_level": severity_report["gaussian"]["level"],
            "poisson_level": severity_report["poisson"]["level"],
            "total_noise": severity_report["summary"]["total_noise_percentage"],
            "total_level": severity_report["summary"]["total_noise_level"],
            "original_url": f"/static/uploads/{file.filename}",
            "annotated_url": f"/static/outputs/{output_filename}",
            "overlay_url": f"/static/outputs/{overlay_filename}",
            "pixels": severity_report["pixels"],
            "regions": visuals.get("regions", [])
        }
        
        # --- MODEL 2 (Attention U-Net) Independent Pipeline ---
        m2_data = None
        if model2_loaded and predictor_m2 is not None:
            try:
                m2_mask = predictor_m2.predict(str(file_path))
                m2_report = calculate_severity_model2(m2_mask)
                m2_visualizer = Model2Visualizer(image)
                stem = Path(file.filename).stem
                m2_output_filename = f"{stem}_m2_result.png"
                m2_output_path = OUTPUTS_M2_DIR / m2_output_filename
                m2_visuals = m2_visualizer.generate_full_visualization(
                    m2_mask, m2_report, str(m2_output_path)
                )
                m2_data = {
                    "model": "Model 2",
                    "architecture": "Attention U-Net",
                    "noise": m2_report["noise"],
                    "summary": m2_report["summary"],
                    "images": {
                        "mask": f"/static/outputs_m2/{stem}_m2_result_mask.png",
                        "overlay": f"/static/outputs_m2/{stem}_m2_result_overlay.png",
                        "annotated": f"/static/outputs_m2/{m2_output_filename}",
                    }
                }
            except Exception as e2:
                print(f"Model 2 Predict Pipeline Error: {e2}")
                m2_data = {"error": str(e2)}

        response_data["model2"] = m2_data
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Prediction Pipeline Error: {e}")
        return jsonify({"error": f"AI inference failed: {str(e)}"}), 500



@app.route("/predict/model2", methods=["POST"])
def predict_model2():
    """
    Standalone Model 2 (Attention U-Net) inference endpoint.
    Runs ONLY the Attention U-Net pipeline — completely independent of Model 1.
    Returns Poisson + Speckle segmentation, severity, and visualization URLs.
    """
    if not model2_loaded or predictor_m2 is None:
        return jsonify({"error": "Model 2 (Attention U-Net) is not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file key found in multipart/form-data"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Selected filename is empty"}), 400

    try:
        file_path = UPLOADS_DIR / file.filename
        file.save(str(file_path))

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Invalid file format or corrupted image."}), 400

        # Model 2 Pipeline
        m2_mask = predictor_m2.predict(str(file_path))
        m2_report = calculate_severity_model2(m2_mask)
        m2_visualizer = Model2Visualizer(image)
        stem = Path(file.filename).stem
        m2_output_filename = f"{stem}_m2_result.png"
        m2_output_path = OUTPUTS_M2_DIR / m2_output_filename
        m2_visualizer.generate_full_visualization(m2_mask, m2_report, str(m2_output_path))

        return jsonify({
            "success": True,
            "model": "Model 2",
            "architecture": "Attention U-Net",
            "filename": file.filename,
            "original_url": f"/static/uploads/{file.filename}",
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
    Standalone Model 2 demo — generates a CT phantom and runs ONLY the Attention U-Net.
    """
    if not model2_loaded or predictor_m2 is None:
        return jsonify({"error": "Model 2 (Attention U-Net) is not loaded."}), 500

    try:
        from scripts.generate_realistic_ct import create_realistic_ct_phantom_model2

        filename = "demo_phantom.png"
        file_path = UPLOADS_DIR / filename
        create_realistic_ct_phantom_model2(str(file_path))

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Failed to read generated demo scan."}), 500

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


# =====================================================================
# MODEL 3: DeepLabV3+ (Salt & Pepper + RVIN) Endpoints
# =====================================================================

@app.route("/predict/model3", methods=["POST"])
def predict_model3():
    """
    Standalone Model 3 (DeepLabV3+) inference endpoint.
    Returns Salt & Pepper and RVIN noise segmentation, severity, and visualization.
    """
    if not model3_loaded or predictor_m3 is None:
        return jsonify({"error": "Model 3 (DeepLabV3+) is not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file key found in multipart/form-data"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Selected filename is empty"}), 400

    try:
        file_path = UPLOADS_DIR / file.filename
        file.save(str(file_path))

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Invalid file format or corrupted image."}), 400

        m3_mask = predictor_m3.predict(str(file_path))
        m3_report = calculate_severity_model3(m3_mask)
        m3_visualizer = Model3Visualizer(image)
        stem = Path(file.filename).stem
        m3_output_filename = f"{stem}_m3_result.png"
        m3_output_path = OUTPUTS_M3_DIR / m3_output_filename
        m3_visualizer.generate_full_visualization(m3_mask, m3_report, str(m3_output_path))

        return jsonify({
            "success": True,
            "model": "Model 3",
            "architecture": "DeepLabV3+",
            "filename": file.filename,
            "original_url": f"/static/uploads/{file.filename}",
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
    if not model3_loaded or predictor_m3 is None:
        return jsonify({"error": "Model 3 (DeepLabV3+) is not loaded."}), 500

    try:
        from scripts.generate_realistic_ct import create_realistic_ct_phantom

        filename = "demo_phantom.png"
        file_path = UPLOADS_DIR / filename
        create_realistic_ct_phantom(str(file_path))

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Failed to read generated demo scan."}), 500

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


# =====================================================================
# MODEL 4: NoiseCNN (Quantization + Periodic Noise) Endpoints
# =====================================================================

@app.route("/predict/model4", methods=["POST"])
def predict_model4():
    """
    Standalone Model 4 (NoiseCNN) inference endpoint.
    Returns Quantization and Periodic noise probability classification and spectrum visualization.
    """
    if not model4_loaded or predictor_m4 is None:
        return jsonify({"error": "Model 4 (NoiseCNN) is not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file key found in multipart/form-data"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Selected filename is empty"}), 400

    try:
        file_path = UPLOADS_DIR / file.filename
        file.save(str(file_path))

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Invalid file format or corrupted image."}), 400

        m4_probs = predictor_m4.predict(str(file_path))
        m4_report = calculate_severity_model4(m4_probs)
        m4_visualizer = Model4Visualizer(image)
        stem = Path(file.filename).stem
        m4_output_filename = f"{stem}_m4_result.png"
        m4_output_path = OUTPUTS_M4_DIR / m4_output_filename
        m4_visualizer.generate_full_visualization(m4_probs, m4_report, str(m4_output_path))

        return jsonify({
            "success": True,
            "model": "Model 4",
            "architecture": "NoiseCNN",
            "filename": file.filename,
            "original_url": f"/static/uploads/{file.filename}",
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
    if not model4_loaded or predictor_m4 is None:
        return jsonify({"error": "Model 4 (NoiseCNN) is not loaded."}), 500

    try:
        from scripts.generate_realistic_ct import create_realistic_ct_phantom

        filename = "demo_phantom.png"
        file_path = UPLOADS_DIR / filename
        create_realistic_ct_phantom(str(file_path))

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({"error": "Failed to read generated demo scan."}), 500

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
            if not model2_loaded or predictor_m2 is None:
                return jsonify({"error": "Model 2 (Attention U-Net) is not loaded."}), 500
            annotated_path = OUTPUTS_M2_DIR / f"{stem}_m2_result.png"
            if not annotated_path.exists():
                annotated_path = OUTPUTS_M2_DIR / f"{stem}_result.png"
            m2_mask = predictor_m2.predict(str(original_path))
            severity_report = calculate_severity_model2(m2_mask)
            
            # If annotated image missing, generate it
            if not annotated_path.exists():
                img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
                Model2Visualizer(img).generate_full_visualization(m2_mask, severity_report, str(annotated_path))

        elif model_key in ["model3", "m3"]:
            if not model3_loaded or predictor_m3 is None:
                return jsonify({"error": "Model 3 (DeepLabV3+) is not loaded."}), 500
            annotated_path = OUTPUTS_M3_DIR / f"{stem}_m3_result.png"
            if not annotated_path.exists():
                annotated_path = OUTPUTS_M3_DIR / f"{stem}_result.png"
            m3_mask = predictor_m3.predict(str(original_path))
            severity_report = calculate_severity_model3(m3_mask)
            
            if not annotated_path.exists():
                img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
                Model3Visualizer(img).generate_full_visualization(m3_mask, severity_report, str(annotated_path))

        elif model_key in ["model4", "m4"]:
            if not model4_loaded or predictor_m4 is None:
                return jsonify({"error": "Model 4 (NoiseCNN) is not loaded."}), 500
            annotated_path = OUTPUTS_M4_DIR / f"{stem}_m4_result.png"
            if not annotated_path.exists():
                annotated_path = OUTPUTS_M4_DIR / f"{stem}_result.png"
            m4_probs = predictor_m4.predict(str(original_path))
            severity_report = calculate_severity_model4(m4_probs)
            
            if not annotated_path.exists():
                img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
                Model4Visualizer(img).generate_full_visualization(m4_probs, severity_report, str(annotated_path))

        else: # Model 1 (U-Net++)
            annotated_path = OUTPUTS_DIR / f"{stem}_result.png"
            predicted_mask = predictor.predict(str(original_path))
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Launching LungCT AI Flask server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)



