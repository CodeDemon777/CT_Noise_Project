# LungCT AI: Multi-Noise Classification & Clinical Reporting System

A professional, full-stack Deep Learning Medical AI platform designed to detect, classify, segment, and quantify noise artifacts in Computed Tomography (CT) scans across four specialized neural architectures.

---

## 🏛️ Project Architecture

```
CT_Noise_Project/
├── frontend/                              # Pure static client (Deployable on Vercel/Netlify/Nginx/Flask)
│   ├── index.html                         # Diagnostic UI dashboard & model viewports
│   ├── css/
│   │   └── style.css                      # Modern responsive glassmorphic styles
│   └── js/
│       └── script.js                      # Multi-model event handlers, sync & PDF downloader
│
├── backend/                               # REST API & Deep Learning Inference Server
│   ├── app.py                             # Main Flask application & dynamic routing
│   ├── model_loader.py                    # Multi-model weight & checkpoint loader
│   ├── predict.py                         # Model 1 (U-Net++) inference pipeline
│   ├── severity.py                        # Model 1 severity metric engine
│   ├── visualization.py                   # Model 1 annotation & overlay visualizer
│   ├── report_generator.py                # Multi-model ReportLab clinical PDF generator
│   ├── models/                            # PyTorch model definitions & checkpoints
│   │   ├── model1/                        # U-Net++ (best_model.pth)
│   │   ├── model2/                        # Attention U-Net (Joshna.pth)
│   │   ├── model3/                        # DeepLabV3+ (Jahnavi (1).pth)
│   │   └── model4/                        # NoiseCNN (Vasanth (2).pth)
│   ├── static/                            # Server runtime storage
│   │   ├── uploads/                       # Ingested CT scans
│   │   ├── outputs/                       # Model 1 visualizations & masks
│   │   ├── outputs_m2/                    # Model 2 visualizations & overlays
│   │   ├── outputs_m3/                    # Model 3 visualizations & masks
│   │   ├── outputs_m4/                    # Model 4 visualizations & FFT spectrums
│   │   └── reports/                       # Generated clinical PDF reports
│   └── requirements.txt                   # Backend Python dependencies
│
├── tests/                                 # Automated Test Suites
│   ├── test_model.py                      # Model 1 unit test
│   ├── test_model2.py                     # Model 2 integration test
│   ├── test_model3.py                     # Model 3 unit test
│   ├── test_model4.py                     # Model 4 unit test
│   ├── test_inference.py                  # Integration inference test
│   └── verify_dataset_models.py           # Ground-truth HDF5 dataset verification suite
│
├── scripts/                               # Dataset & CT Phantom Generation Tools
│   ├── generate_dataset_test_images.py    # 3D HDF5 dataset slice extractor & noise injector
│   ├── generate_realistic_ct.py           # Synthetic CT phantom generator
│   └── predict_single.py                  # Single-image CLI prediction tool
│
├── .gitignore                             # Production ignore rules
├── requirements.txt                       # Root level Python dependencies
├── Procfile                               # Render / Heroku deployment descriptor
├── Dockerfile                             # Containerized hosting file
└── README.md                              # Project documentation
```

---

## 🔬 Deep Learning Architectures

| Model | Architecture | Checkpoint | Target Noise Artifacts | Output Format |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1** | **U-Net++ (Nested UNet)** | `best_model.pth` | **Gaussian Noise** & **Poisson Noise** | 3-class pixel segmentation mask (Dice: 0.9886) |
| **Model 2** | **Attention U-Net** | `Joshna.pth` | **Poisson Noise** & **Speckle Noise** | Multi-channel attention segmentation mask |
| **Model 3** | **DeepLabV3+** | `Jahnavi (1).pth` | **Salt & Pepper** & **RVIN Noise** | Atrous Spatial Pyramid Pooling (ASPP) mask |
| **Model 4** | **NoiseCNN** | `Vasanth (2).pth` | **Quantization** & **Periodic Noise** | Softmax probability & 2D FFT Magnitude Spectrum |

---

## 🚀 Quickstart & Local Hosting

### 1. Prerequisites
- Python 3.10+
- PyTorch (CPU or CUDA GPU)

### 2. Installation
```bash
# Clone or navigate to the repository
cd CT_Noise_Project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Server
```bash
# Start backend and serve frontend
python backend/app.py
```
Open **`http://localhost:8080`** in your browser.

---

## ☁️ Cloud Deployment Options

### Option A: Render / Railway / Heroku
The repository contains a [`Procfile`](Procfile) ready for cloud platform deployment:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT backend.app:app`

### Option B: Docker Containerization
```bash
# Build Docker image
docker build -t ct-noise-ai .

# Run Docker container
docker run -p 8080:8080 ct-noise-ai
```

### Option C: Decoupled Hosting (Frontend on Vercel/Netlify, Backend on Cloud)
- **Frontend**: Deploy the `frontend/` directory to Vercel, Netlify, or GitHub Pages.
- **Backend**: Deploy the `backend/` directory to Render / Railway / AWS.
- Set the API URL in `frontend/js/script.js` if hosting backend on a separate domain.

---

## 🧪 Testing & Quality Assurance

Run all test suites:
```bash
# Run all unit tests
pytest tests/

# Run ground-truth HDF5 dataset verification harness
python tests/verify_dataset_models.py
```

---

## 📄 License & Disclaimer
Developed as a Software Engineering Capstone Project for research and educational purposes. Model outputs should be cross-referenced with medical imaging hardware.