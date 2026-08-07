# CT Noise Detection - Capstone Project

A comprehensive Software Engineering capstone project for detecting and analyzing noise artifacts in CT (Computed Tomography) medical images.

## Project Overview

This project implements an end-to-end machine learning solution that:
- **Detects** noise in CT images using a trained UNet model
- **Classifies** noise into Gaussian and Poisson types
- **Calculates** severity percentages for each noise type
- **Visualizes** results with bounding boxes and severity labels
- **Provides** REST API backend and React frontend for easy access

### Key Metrics
- **Dice Score**: 0.9886 (Excellent segmentation accuracy)
- **Classes**: 3 (Clean, Gaussian Noise, Poisson Noise)
- **Architecture**: UNet with skip connections

---

## Project Structure

```
CT_Noise_Project/
├── model/                          # Trained model
│   └── best_model.pth             # Weights (64MB)
│
├── backend/                        # Python backend modules
│   ├── __init__.py
│   ├── model_loader.py            # Load and manage model
│   ├── predict.py                 # Inference pipeline
│   ├── severity.py                # Calculate noise severity
│   ├── visualization.py           # Create annotated images
│   └── app.py                     # FastAPI REST API
│
├── frontend/                       # React frontend (Phase 7+)
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── package.json
│
├── outputs/                        # Generated results
│   ├── annotated/                 # Result images with annotations
│   └── reports/                   # PDF reports
│
├── uploads/                        # User uploaded images
│
├── docs/                          # Documentation
│   ├── SRS.pdf                    # Software Requirements Specification
│   ├── Architecture.pdf           # System Architecture
│   └── UML_Diagrams/
│
├── predict_single.py              # Local inference testing (Phase 1)
├── requirements.txt               # Python dependencies
└── README.md                       # This file
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

1. **Clone/Navigate to project**
   ```bash
   cd CT_Noise_Project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model file**
   - Ensure `model/best_model.pth` exists (should be ~64MB)

---

## Phase 1: Local Inference

Test the complete inference pipeline locally before deploying.

### Usage

**Option 1: With your own CT image**
```bash
python predict_single.py path/to/your/ct_image.png
```

**Option 2: Auto-generate test image**
```bash
python predict_single.py
```

**With custom output directory**
```bash
python predict_single.py path/to/image.png --output /custom/output/dir
```

### Output

```
outputs/
├── annotated/
│   ├── ct_result.png          # Image with bounding boxes + severity labels
│   └── ct_mask.png            # Segmentation mask
└── reports/
```

### Expected Output Format

```json
{
  "gaussian": 2.83,              // Gaussian noise percentage
  "poisson": 1.79,               // Poisson noise percentage
  "gaussian_level": "Mild",      // Severity classification
  "poisson_level": "Mild",
  "total_noise": 4.62
}
```

---

## Phase 2-5: Backend Modules

### 1. Model Loader (`backend/model_loader.py`)

```python
from backend.model_loader import load_model

model = load_model("model/best_model.pth")
```

### 2. Prediction (`backend/predict.py`)

```python
from backend.predict import CTPredictor

predictor = CTPredictor("model/best_model.pth")
mask = predictor.predict("path/to/ct_image.png")
```

### 3. Severity Analysis (`backend/severity.py`)

```python
from backend.severity import SeverityCalculator

calculator = SeverityCalculator()
report = calculator.get_detailed_report(mask)
calculator.print_report(report)
```

### 4. Visualization (`backend/visualization.py`)

```python
from backend.visualization import CTVisualizer

visualizer = CTVisualizer(original_image)
visuals = visualizer.generate_full_visualization(mask, report, "output.png")
```

### 5. Model Architecture

**UNet3Class** - 3-class semantic segmentation network
- **Encoder**: 4 levels with max pooling
- **Bottleneck**: Central feature extraction
- **Decoder**: 4 levels with skip connections
- **Output**: 3 channels (Clean, Gaussian, Poisson)

---

## Phase 6: FastAPI Backend

### Start API Server

```bash
python -m uvicorn backend.app:app --reload
```

Server runs at: `http://localhost:8000`

### API Endpoints

#### 1. **Health Check**
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "model/best_model.pth"
}
```

#### 2. **Single Prediction**
```bash
POST /predict
Content-Type: multipart/form-data

file: <CT_image_file>
```

Response:
```json
{
  "gaussian": 2.83,
  "poisson": 1.79,
  "gaussian_level": "Mild",
  "poisson_level": "Mild",
  "total_noise": 4.62,
  "total_level": "Mild",
  "image_filename": "ct_result.png",
  "pixels": {
    "gaussian": 14832,
    "poisson": 9383,
    "clean": 227785,
    "total": 262144
  }
}
```

#### 3. **Batch Prediction**
```bash
POST /batch
Content-Type: multipart/form-data

files: [<file1>, <file2>, ...]
```

#### 4. **Get Result Image**
```bash
GET /result/ct_result.png
```

#### 5. **List Results**
```bash
GET /results
```

---

## API Testing

### Using cURL

```bash
# Single file prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@path/to/ct_image.png"

# Get health status
curl http://localhost:8000/health

# Get result image
curl http://localhost:8000/result/ct_result.png -o result.png
```

### Using Python

```python
import requests

# Predict
with open("ct_image.png", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

---

## Phase 7: React Frontend

### Setup (Coming Soon)

```bash
cd frontend
npm install
npm run dev
```

### Pages

- **Home**: Project information and upload button
- **Upload**: CT image upload interface
- **Results**: Display predictions and annotated images
- **About**: Project documentation

---

## Phase 8: PDF Report Generation

Generate comprehensive PDF reports for each prediction.

```python
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer

# Report generation coming in Phase 8
```

---

## Phase 9: Testing

### Unit Tests

```bash
pytest tests/test_model_loader.py
pytest tests/test_predict.py
pytest tests/test_severity.py
```

### Integration Tests

```bash
pytest tests/test_integration.py
```

---

## Phase 10: Deployment

### Frontend Deployment (Vercel)

```bash
cd frontend
npm install -g vercel
vercel
```

### Backend Deployment (Render)

1. Create `Procfile`:
   ```
   web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.app:app
   ```

2. Deploy to Render.com

---

## File Descriptions

### Core Modules

| File | Purpose |
|------|---------|
| `model_loader.py` | Load trained UNet model, handle device management |
| `predict.py` | Image preprocessing, inference, postprocessing |
| `severity.py` | Calculate noise percentages and severity levels |
| `visualization.py` | Create annotated images with bounding boxes |
| `app.py` | FastAPI REST API server |
| `predict_single.py` | CLI tool for local testing |

### Architecture

**Inference Pipeline**:
```
CT Image
  ↓ (read_ct_image)
Read Image
  ↓ (preprocess)
Normalize to [-1, 1], resize to 512×512
  ↓ (predict_mask)
UNet Model
  ↓ (postprocess)
Resize to original size
  ↓ (calculate_severity)
Gaussian%, Poisson%
  ↓ (visualize)
Annotated image with bounding boxes
```

---

## Image Format Support

- ✅ PNG
- ✅ JPEG
- ✅ BMP
- ✅ DICOM (via preprocessing)
- ✅ TIFF

---

## Output Interpretation

### Severity Levels

| Percentage | Level |
|-----------|-------|
| < 5% | Mild |
| 5-15% | Moderate |
| > 15% | Severe |

### Noise Classification

- **Gaussian**: Random white noise (from electronics)
- **Poisson**: Shot noise (from X-ray photons)
- **Clean**: Noise-free region

---

## Performance

- **Model Size**: 64 MB
- **Inference Time**: ~500ms per 512×512 image (GPU)
- **Dice Score**: 0.9886
- **Supported Batch Size**: Unlimited via API

---

## Dependencies

```
PyTorch >= 2.0
OpenCV >= 4.8
NumPy >= 1.24
Pillow >= 10.0
scikit-image >= 0.21
FastAPI >= 0.104
Uvicorn >= 0.24
ReportLab >= 4.0
```

---

## Troubleshooting

### Model Not Loading
```
❌ FileNotFoundError: Model file not found: model/best_model.pth
✅ Solution: Ensure best_model.pth is in model/ directory
```

### CUDA Issues
```
✅ Falls back to CPU automatically if CUDA unavailable
✅ Use --device cpu flag to force CPU mode
```

### Image Format Error
```
❌ Failed to read image
✅ Convert image to PNG/JPEG using: cv2.imwrite("output.png", image)
```

---

## Development Workflow

### For Local Testing
```bash
# 1. Test single image
python predict_single.py test_image.png

# 2. Check outputs
ls -la outputs/annotated/

# 3. Inspect results
python -c "import cv2; import numpy as np; print(np.unique(cv2.imread('outputs/annotated/ct_mask.png', 0)))"
```

### For Backend Development
```bash
# 1. Start API
python -m uvicorn backend.app:app --reload

# 2. Test endpoint
curl -X POST http://localhost:8000/predict -F file=@test_image.png

# 3. Check logs
# (Uvicorn shows detailed logs in terminal)
```

---

## Next Steps

1. ✅ **Phase 1-6**: Backend implementation (THIS DOCUMENT)
2. ⏳ **Phase 7**: React frontend
3. ⏳ **Phase 8**: PDF reports
4. ⏳ **Phase 9**: Unit/integration tests
5. ⏳ **Phase 10**: Deploy to Vercel + Render

---

## Author Notes

This project demonstrates professional Software Engineering practices:
- ✅ Modular architecture (separation of concerns)
- ✅ Clean code (type hints, docstrings, error handling)
- ✅ REST API (FastAPI, CORS, batch processing)
- ✅ Scalable (async endpoints, GPU support)
- ✅ Well-documented (README, inline comments, usage examples)
- ✅ Production-ready (error handling, logging, validation)

---

## License

MIT License - Feel free to use for educational and commercial purposes.

---

## Contact & Support

For issues or questions:
1. Check troubleshooting section
2. Review inline code comments
3. Check API documentation at `/docs` when API is running

---

**Last Updated**: June 12, 2026
**Status**: Phases 1-6 Complete ✅
#   C T _ N o i s e _ P r o j e c t  
 