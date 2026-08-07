"""
Flask Backend for LungCT AI Capstone Project
Serves the web dashboard and handles REST API requests for upload, prediction, and report download.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

# Re-configure standard streams for UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Setup project root for local module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.predict import CTPredictor
from backend.severity import SeverityCalculator
from backend.visualization import CTVisualizer
from backend.report_generator import CTReportGenerator

# Initialize Flask app serving the templates and static files from project root
app = Flask(
    __name__,
    static_folder=str(PROJECT_ROOT / "static"),
    template_folder=str(PROJECT_ROOT / "templates")
)

# Enable Cross-Origin Resource Sharing
CORS(app)

MODEL_PATH = PROJECT_ROOT / "model" / "best_model.pth"
STATIC_DIR = PROJECT_ROOT / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
OUTPUTS_DIR = STATIC_DIR / "outputs"
REPORTS_DIR = STATIC_DIR / "reports"

# Ensure all server storage directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load U-Net++ predictor on startup
try:
    predictor = CTPredictor(str(MODEL_PATH))
    model_loaded = True
    print(f"✅ U-Net++ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model_loaded = False


@app.route("/")
def index():
    """
    Renders the main clinical dashboard landing page.
    """
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    API Health check endpoint returning model deployment status.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH)
    })


@app.route("/demo", methods=["GET"])
def demo():
    """
    Generates a realistic CT scan phantom with noise, runs prediction, and returns analysis.
    Allows testing the app immediately without uploading an external file.
    """
    if not model_loaded:
        return jsonify({"error": "Inference server state error: U-Net++ model is not loaded."}), 500
        
    try:
        from generate_realistic_ct import create_realistic_ct_phantom
        
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
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Prediction Pipeline Error: {e}")
        return jsonify({"error": f"AI inference failed: {str(e)}"}), 500


@app.route("/report", methods=["GET"])
def report():
    """
    Generate downloadable PDF clinical report for a diagnosed image.
    Loads original, result annotations, computes metrics and writes reportlab document.
    """
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "Missing required query parameter: filename"}), 400
        
    original_path = UPLOADS_DIR / filename
    if not original_path.exists():
        return jsonify({"error": "Requested original CT scan was not found on server."}), 404
        
    stem = Path(filename).stem
    annotated_filename = f"{stem}_result.png"
    annotated_path = OUTPUTS_DIR / annotated_filename
    
    if not annotated_path.exists():
        return jsonify({"error": "Diagnostic analysis has not been performed on this scan yet."}), 400
        
    pdf_filename = f"{stem}_report.pdf"
    pdf_path = REPORTS_DIR / pdf_filename
    
    try:
        # Run local prediction to generate the mask and detailed statistics
        predicted_mask = predictor.predict(str(original_path))
        calculator = SeverityCalculator()
        severity_report = calculator.get_detailed_report(predicted_mask)
        
        # Build ReportLab PDF
        CTReportGenerator.generate_pdf(
            output_pdf_path=str(pdf_path),
            original_img_path=str(original_path),
            annotated_img_path=str(annotated_path),
            report_data=severity_report,
            filename=filename
        )
        
        return send_file(str(pdf_path), as_attachment=True, download_name=pdf_filename)
        
    except Exception as e:
        print(f"Report Generation Error: {e}")
        return jsonify({"error": f"Failed to generate clinical PDF: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Launching LungCT AI Flask server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

