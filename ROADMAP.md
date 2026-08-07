# Development Roadmap - Phases 7-10

**Current Status**: Phases 1-6 ✅ Complete  
**Next Phase**: Phase 7 - React Frontend

---

## Phase 7: React Frontend

### Objective
Build user interface for uploading CT images and viewing results.

### File Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── ImageUploader.jsx      # File upload component
│   │   ├── ResultDisplay.jsx      # Show results
│   │   ├── SeverityChart.jsx      # Visualize severity
│   │   └── Header.jsx             # Navigation
│   │
│   ├── pages/
│   │   ├── Home.jsx               # Landing page
│   │   ├── Upload.jsx             # Upload interface
│   │   ├── Results.jsx            # Results display
│   │   └── About.jsx              # Project info
│   │
│   ├── App.jsx
│   ├── main.jsx
│   └── App.css
│
├── vite.config.js
├── package.json
└── index.html
```

### Technology Stack
- **Framework**: React 18
- **Build Tool**: Vite
- **HTTP Client**: Axios
- **Styling**: TailwindCSS
- **Charts**: Chart.js or Recharts

### Key Components

#### 1. ImageUploader
```jsx
// Upload CT image
// - Drag & drop support
// - File validation
// - Progress indicator
```

#### 2. ResultDisplay
```jsx
// Show prediction results
// - Original image
// - Annotated image
// - Severity percentages
```

#### 3. SeverityChart
```jsx
// Visualize noise breakdown
// - Pie chart: Gaussian vs Poisson
// - Bar chart: Severity levels
```

### Setup Commands
```bash
cd frontend
npm create vite@latest . -- --template react
npm install axios tailwindcss chart.js react-chartjs-2
npm run dev
```

### API Integration
```javascript
// Call backend endpoints
const response = await axios.post(
  'http://localhost:8000/predict',
  formData
);

const { gaussian, poisson, image_filename } = response.data;
```

---

## Phase 8: PDF Report Generation

### Objective
Generate downloadable PDF reports with predictions and images.

### Dependencies
```bash
pip install reportlab pillow
```

### Implementation

#### ReportLab PDF Generator
```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from datetime import datetime

def generate_report(mask, severity_report, image_path, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    
    # Title
    elements.append(Paragraph("CT NOISE ANALYSIS REPORT", styles['Title']))
    
    # Metadata
    elements.append(Paragraph(f"Date: {datetime.now()}", styles['Normal']))
    
    # Results
    elements.append(Paragraph("Severity Analysis", styles['Heading2']))
    elements.append(Paragraph(f"Gaussian: {severity_report['gaussian']['percentage']}%", styles['Normal']))
    elements.append(Paragraph(f"Poisson: {severity_report['poisson']['percentage']}%", styles['Normal']))
    
    # Image
    elements.append(Image(image_path, width=6*inch, height=4*inch))
    
    doc.build(elements)
```

### Output Structure
```
outputs/reports/
├── report_2026_06_12_001.pdf
├── report_2026_06_12_002.pdf
└── report_2026_06_12_003.pdf
```

### FastAPI Integration
```python
@app.post("/report")
async def generate_report(file: UploadFile):
    # Run prediction
    # Generate report
    return FileResponse("report.pdf", media_type="application/pdf")
```

---

## Phase 9: Testing

### Unit Tests

#### Test Model Loader
```python
# tests/test_model_loader.py
def test_load_model():
    model = load_model("model/best_model.pth")
    assert model is not None
    assert str(model.device) in ['cpu', 'cuda:0']
```

#### Test Prediction
```python
# tests/test_predict.py
def test_predict():
    predictor = CTPredictor("model/best_model.pth")
    mask = predictor.predict("test_image.png")
    assert mask.shape == (512, 512)
    assert set(mask.flatten()) <= {0, 1, 2}
```

#### Test Severity
```python
# tests/test_severity.py
def test_calculate_severity():
    calculator = SeverityCalculator()
    report = calculator.get_detailed_report(test_mask)
    
    assert 'gaussian' in report
    assert 'poisson' in report
    assert 0 <= report['gaussian']['percentage'] <= 100
```

#### Test API
```python
# tests/test_api.py
def test_predict_endpoint():
    with open("test_image.png", "rb") as f:
        response = client.post("/predict", files={"file": f})
    assert response.status_code == 200
    assert "gaussian" in response.json()
```

### Run Tests
```bash
pytest tests/
pytest tests/ -v  # Verbose
pytest tests/ --cov  # Coverage report
```

### Test Coverage Goal
- Target: > 80% code coverage
- Commands:
  ```bash
  pytest --cov=backend tests/
  pytest --cov-report=html tests/
  ```

---

## Phase 10: Deployment

### A. Frontend Deployment (Vercel)

**Step 1: Build**
```bash
cd frontend
npm run build
```

**Step 2: Deploy**
```bash
npm install -g vercel
vercel
```

**Step 3: Environment Variables**
```
VITE_API_URL=https://ct-noise-api.herokuapp.com
```

### B. Backend Deployment (Render.com)

**Step 1: Create Procfile**
```
web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.app:app
```

**Step 2: Create render.yaml**
```yaml
services:
  - type: web
    name: ct-noise-api
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.10
```

**Step 3: Connect GitHub**
- Push code to GitHub
- Link Render.com to repository
- Auto-deploy on push

### Deployment Checklist

**Backend**
- [ ] Create Procfile
- [ ] Update requirements.txt with gunicorn
- [ ] Set CORS allowed origins
- [ ] Test API endpoints
- [ ] Deploy to Render
- [ ] Verify health check

**Frontend**
- [ ] Build optimization
- [ ] Set API URL to production
- [ ] Environment variables
- [ ] Deploy to Vercel
- [ ] Test all pages
- [ ] SSL certificate (auto)

### Production Considerations

#### Environment Variables
```python
# .env (backend)
MODEL_PATH=model/best_model.pth
MAX_UPLOAD_SIZE=50MB
API_CORS_ORIGINS=["https://app-domain.com"]
LOG_LEVEL=info
```

#### Monitoring
- FastAPI middleware for logging
- Error tracking (Sentry integration)
- Performance monitoring

#### Security
- CORS whitelist
- File upload validation
- Rate limiting
- HTTPS enforcement

---

## Technology Stack Summary

### Backend
- **Language**: Python 3.8+
- **Framework**: FastAPI + Uvicorn
- **ML**: PyTorch
- **Image Processing**: OpenCV, scikit-image
- **PDF**: ReportLab
- **Testing**: pytest

### Frontend
- **Language**: JavaScript (ES6+)
- **Framework**: React 18
- **Build**: Vite
- **Styling**: TailwindCSS
- **HTTP**: Axios

### Deployment
- **Backend**: Render.com (Python/Uvicorn)
- **Frontend**: Vercel (Node.js/Static)
- **Database**: Optional (not needed for MVP)

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| 1-6 | ✅ Complete | Done |
| 7 | 3-5 days | Next |
| 8 | 1-2 days | After 7 |
| 9 | 2-3 days | After 8 |
| 10 | 2-3 days | Final |

**Total**: ~11-16 days from start to full deployment

---

## Key Files to Create/Modify

### Phase 7
- `frontend/src/components/ImageUploader.jsx`
- `frontend/src/pages/Upload.jsx`
- `frontend/src/pages/Results.jsx`

### Phase 8
- `backend/reports.py` (new)
- Update `backend/app.py` (add `/report` endpoint)

### Phase 9
- `tests/test_model_loader.py`
- `tests/test_predict.py`
- `tests/test_api.py`

### Phase 10
- `Procfile`
- `render.yaml`
- `.env.example`

---

## Git Workflow Recommendation

```bash
# Create branches for each phase
git checkout -b phase/frontend
git checkout -b phase/testing
git checkout -b phase/deployment

# Commit frequently
git add .
git commit -m "Phase 7: Add image uploader component"
git push origin phase/frontend

# Create pull requests for review
# Merge to main after testing
```

---

## Additional Enhancements (Optional)

### Advanced Features
- [ ] User authentication
- [ ] Image history per user
- [ ] Batch processing dashboard
- [ ] Real-time progress with WebSockets
- [ ] Multi-model comparison
- [ ] Advanced filtering options

### Performance Optimizations
- [ ] Model quantization (reduce size)
- [ ] ONNX conversion (faster inference)
- [ ] Redis caching for results
- [ ] Image preprocessing optimization

### DevOps
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Database integration (PostgreSQL)
- [ ] Admin dashboard

---

## References

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **React Docs**: https://react.dev/
- **Render Docs**: https://render.com/docs
- **Vercel Docs**: https://vercel.com/docs
- **PyTorch Docs**: https://pytorch.org/docs/

---

## Questions to Address in Phase 7+

1. **Frontend**: How to handle large image uploads?
   - Answer: Implement chunked uploads, client-side compression

2. **Storage**: Where to store results?
   - Answer: Temporary in `/outputs`, long-term in cloud storage (AWS S3)

3. **Scaling**: What if thousands of users upload simultaneously?
   - Answer: Use message queue (Celery), GPU cluster (Ray)

4. **Cost**: How much will deployment cost?
   - Answer: ~$10-20/month for starter tier (scales as needed)

---

**Last Updated**: June 12, 2026  
**Next Checkpoint**: Phase 7 Kickoff

---

Ready to build Phase 7? Start with: `npm create vite@latest frontend -- --template react` 🚀
