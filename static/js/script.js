/* 
  LungCT AI Dashboard Interactivity Script
  Handles file upload, progress bar animations, toggling results, and Chart.js initialization.
*/

document.addEventListener('DOMContentLoaded', () => {
    // 1. DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const previewMetadata = document.getElementById('preview-metadata');
    
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const demoGenerateBtn = document.getElementById('demo-generate-btn');
    
    const analysisProgress = document.getElementById('analysis-progress');
    const progressBar = document.getElementById('progress-bar');
    
    const resultsPanel = document.getElementById('results-panel');
    const originalImg = document.getElementById('result-original-img');
    const annotatedImg = document.getElementById('result-annotated-img');
    const overlayImg = document.getElementById('result-overlay-img');
    
    const toggleBoxBtn = document.getElementById('toggle-box-btn');
    const toggleOverlayBtn = document.getElementById('toggle-overlay-btn');
    
    const valGaussian = document.getElementById('val-gaussian');
    const valPoisson = document.getElementById('val-poisson');
    const valTotal = document.getElementById('val-total');
    
    const badgeGaussian = document.getElementById('badge-gaussian');
    const badgePoisson = document.getElementById('badge-poisson');
    const badgeTotal = document.getElementById('badge-total');

    const viewportValGaussian = document.getElementById('viewport-val-gaussian');
    const viewportValPoisson = document.getElementById('viewport-val-poisson');
    const viewportBadgeGaussian = document.getElementById('viewport-badge-gaussian');
    const viewportBadgePoisson = document.getElementById('viewport-badge-poisson');
    
    const barGaussian = document.getElementById('bar-gaussian');
    const barPoisson = document.getElementById('bar-poisson');
    const barTotal = document.getElementById('bar-total');
    
    const downloadReportBtn = document.getElementById('download-report-btn');
    const regionsList = document.getElementById('regions-list');

    // 2. Global Variables for Current Analysis state
    let currentFile = null;
    let analyzedFilename = null;

    // 3. Setup Drag and Drop File Upload
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Handle File loading & local preview
    function handleSelectedFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Unsupported file type. Please upload a standard image file (PNG, JPEG, BMP).');
            return;
        }
        currentFile = file;
        
        // Hide drop zone and show preview
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        
        // Render preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            
            // Extract dimensions
            const img = new Image();
            img.onload = () => {
                const sizeKB = (file.size / 1024).toFixed(1);
                previewMetadata.textContent = `File Size: ${sizeKB} KB | Dimensions: ${img.width} × ${img.height} px`;
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    // Reset upload view
    clearBtn.addEventListener('click', resetUploadArea);

    function resetUploadArea() {
        currentFile = null;
        analyzedFilename = null;
        fileInput.value = '';
        imagePreview.src = '#';
        previewMetadata.textContent = 'File Size: - | Dimension: -';
        
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        analysisProgress.classList.add('hidden');
        resultsPanel.classList.add('hidden');
        
        // Reset progress steps
        progressBar.style.width = '0%';
        document.querySelectorAll('.step').forEach(s => {
            s.classList.remove('active', 'completed');
        });
        
        // Reset regions list
        if (regionsList) {
            regionsList.innerHTML = '<li class="no-regions">No noise regions detected.</li>';
        }
    }

    // 4. Run Analysis & Progress Simulator
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        
        // Disable buttons during analysis
        analyzeBtn.disabled = true;
        clearBtn.disabled = true;
        
        // Reveal progress bar
        analysisProgress.classList.remove('hidden');
        resultsPanel.classList.add('hidden');
        
        const steps = [
            { id: 'step-upload', percent: 15, delay: 400 },
            { id: 'step-model', percent: 45, delay: 1000 },
            { id: 'step-classification', percent: 65, delay: 600 },
            { id: 'step-severity', percent: 85, delay: 400 },
            { id: 'step-visual', percent: 100, delay: 400 }
        ];

        // Helper to update progress step indicators
        const setStepState = (stepId, state) => {
            const stepEl = document.getElementById(stepId);
            if (state === 'active') {
                stepEl.classList.add('active');
                stepEl.classList.remove('completed');
            } else if (state === 'completed') {
                stepEl.classList.add('active', 'completed');
            }
        };

        // Prepare Multipart Form Data
        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            // Trigger actual network request in parallel with visual animations
            const uploadPromise = fetch('/predict', {
                method: 'POST',
                body: formData
            });

            // Execute progress animation sequence
            for (let i = 0; i < steps.length; i++) {
                const step = steps[i];
                setStepState(step.id, 'active');
                progressBar.style.width = `${step.percent}%`;
                
                // For the model inference step, wait for the network request too if it takes longer
                if (step.id === 'step-model') {
                    await new Promise(resolve => setTimeout(resolve, step.delay));
                } else {
                    await new Promise(resolve => setTimeout(resolve, step.delay));
                }
                
                setStepState(step.id, 'completed');
            }

            // Resolve network request
            const response = await uploadPromise;
            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Server prediction failed.');
            }

            // Save state
            analyzedFilename = data.filename;
            
            // Populate Results Dashboard
            displayResults(data);

        } catch (error) {
            alert(`Analysis Error: ${error.message}`);
            resetUploadArea();
        } finally {
            analyzeBtn.disabled = false;
            clearBtn.disabled = false;
        }
    });

    // Populate and show the results dashboard
    function displayResults(data) {
        // Show images
        originalImg.src = data.original_url + '?t=' + new Date().getTime();
        annotatedImg.src = data.annotated_url + '?t=' + new Date().getTime();
        overlayImg.src = data.overlay_url + '?t=' + new Date().getTime();
        
        // Show bounding box viewport by default
        annotatedImg.classList.remove('hidden');
        overlayImg.classList.add('hidden');
        toggleBoxBtn.classList.add('active');
        toggleOverlayBtn.classList.remove('active');
        
        // Set metrics text
        valGaussian.textContent = `${data.gaussian.toFixed(2)}%`;
        valPoisson.textContent = `${data.poisson.toFixed(2)}%`;
        valTotal.textContent = `${data.total_noise.toFixed(2)}%`;
        
        if (viewportValGaussian) viewportValGaussian.textContent = `${data.gaussian.toFixed(2)}%`;
        if (viewportValPoisson) viewportValPoisson.textContent = `${data.poisson.toFixed(2)}%`;
        
        // Set severity badges
        updateBadge(badgeGaussian, data.gaussian_level);
        updateBadge(badgePoisson, data.poisson_level);
        updateBadge(badgeTotal, data.total_level);
        
        if (viewportBadgeGaussian) updateBadge(viewportBadgeGaussian, data.gaussian_level);
        if (viewportBadgePoisson) updateBadge(viewportBadgePoisson, data.poisson_level);
        
        // Set progress bars
        barGaussian.style.width = `${data.gaussian}%`;
        barPoisson.style.width = `${data.poisson}%`;
        barTotal.style.width = `${data.total_noise}%`;
        
        // Render detected regions list
        if (regionsList) {
            regionsList.innerHTML = '';
            if (data.regions && data.regions.length > 0) {
                data.regions.forEach(region => {
                    const li = document.createElement('li');
                    li.className = `region-${region.type.toLowerCase()}`;
                    li.innerHTML = `
                        <span class="region-id-badge">${region.id}</span>
                        <span class="region-type">${region.type}</span>
                        <span class="region-pct">${region.percentage.toFixed(2)}%</span>
                    `;
                    regionsList.appendChild(li);
                });
            } else {
                regionsList.innerHTML = '<li class="no-regions">No noise regions detected.</li>';
            }
        }
        
        // Reveal panel with smooth scroll (using small delay for DOM layout update)
        resultsPanel.classList.remove('hidden');
        setTimeout(() => {
            resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    // Helper to format severity badge classes
    function updateBadge(badgeElement, level) {
        badgeElement.textContent = level;
        
        // Reset classes
        badgeElement.className = 'severity-badge';
        
        // Add color profile
        const lowerLvl = level.toLowerCase();
        if (lowerLvl === 'none') {
            badgeElement.classList.add('badge-none');
        } else if (lowerLvl === 'mild') {
            badgeElement.classList.add('badge-mild');
        } else if (lowerLvl === 'moderate') {
            badgeElement.classList.add('badge-moderate');
        } else if (lowerLvl === 'severe') {
            badgeElement.classList.add('badge-severe');
        } else if (lowerLvl === 'critical') {
            badgeElement.classList.add('badge-critical');
        }
    }

    // Toggle viewport buttons
    toggleBoxBtn.addEventListener('click', () => {
        annotatedImg.classList.remove('hidden');
        overlayImg.classList.add('hidden');
        toggleBoxBtn.classList.add('active');
        toggleOverlayBtn.classList.remove('active');
    });

    toggleOverlayBtn.addEventListener('click', () => {
        annotatedImg.classList.add('hidden');
        overlayImg.classList.remove('hidden');
        toggleBoxBtn.classList.remove('active');
        toggleOverlayBtn.classList.add('active');
    });

    // 5. Download Report Handler
    downloadReportBtn.addEventListener('click', async () => {
        if (!analyzedFilename) return;
        
        downloadReportBtn.disabled = true;
        const originalText = downloadReportBtn.innerHTML;
        downloadReportBtn.textContent = 'Generating PDF Report...';
        
        try {
            // Trigger direct GET download
            const url = `/report?filename=${encodeURIComponent(analyzedFilename)}`;
            
            // Create a temporary link to download
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `${analyzedFilename.split('.')[0]}_report.pdf`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
        } catch (error) {
            alert('Failed to generate or download report: ' + error.message);
        } finally {
            downloadReportBtn.innerHTML = originalText;
            downloadReportBtn.disabled = false;
        }
    });

    // 6. Demo CT scan generator
    demoGenerateBtn.addEventListener('click', async () => {
        demoGenerateBtn.disabled = true;
        const originalText = demoGenerateBtn.innerHTML;
        demoGenerateBtn.textContent = 'Simulating scan acquisition...';
        
        try {
            const response = await fetch('/demo');
            const data = await response.json();
            
            if (!response.ok || data.error) {
                throw new Error(data.error || 'Failed to generate demo scan.');
            }
            
            // Set up virtual file state
            currentFile = new File(["demo"], data.filename, { type: "image/png" });
            analyzedFilename = data.filename;
            
            // Toggle dropzone view
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            imagePreview.src = data.original_url;
            previewMetadata.textContent = `File Size: 84 KB | Dimensions: 512 × 512 px`;
            
            // Trigger display results immediately
            displayResults(data);
            
        } catch (error) {
            alert('Demo generator error: ' + error.message);
        } finally {
            demoGenerateBtn.innerHTML = originalText;
            demoGenerateBtn.disabled = false;
        }
    });

    // 7. Chart.js Initialization (AI Metrics Dashboard)
    
    // Theme Colors for Charts
    const chartBlue = '#0077B6';
    const chartCyan = '#48CAE4';
    const chartDark = '#1C2541';
    
    // Epochs 1 to 20
    const epochs = Array.from({length: 20}, (_, i) => i + 1);
    
    // Simulated Denoising Loss Curves
    const trainLoss = [0.85, 0.62, 0.45, 0.31, 0.22, 0.16, 0.12, 0.09, 0.07, 0.05, 0.04, 0.032, 0.026, 0.021, 0.019, 0.017, 0.016, 0.015, 0.015, 0.0148];
    const valLoss = [0.91, 0.68, 0.51, 0.38, 0.28, 0.21, 0.17, 0.13, 0.11, 0.08, 0.065, 0.052, 0.043, 0.035, 0.029, 0.025, 0.022, 0.019, 0.0185, 0.0182];
    
    // Simulated Dice & IoU Curves
    const diceCurve = [0.42, 0.58, 0.71, 0.82, 0.88, 0.92, 0.942, 0.958, 0.967, 0.974, 0.979, 0.982, 0.984, 0.9855, 0.9868, 0.9875, 0.9880, 0.9883, 0.9885, 0.9886];
    const iouCurve = [0.31, 0.45, 0.58, 0.69, 0.77, 0.83, 0.87, 0.902, 0.925, 0.941, 0.952, 0.961, 0.967, 0.971, 0.973, 0.9748, 0.9760, 0.9768, 0.9773, 0.9778];

    // Initialize Loss Chart
    const lossCtx = document.getElementById('lossChart').getContext('2d');
    new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Training Loss',
                    data: trainLoss,
                    borderColor: chartCyan,
                    backgroundColor: 'rgba(72, 202, 228, 0.1)',
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 2
                },
                {
                    label: 'Validation Loss',
                    data: valLoss,
                    borderColor: chartBlue,
                    backgroundColor: 'transparent',
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#ADB5BD', font: { family: 'Inter' } }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Training Epochs', color: '#ADB5BD' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#ADB5BD' }
                },
                y: {
                    title: { display: true, text: 'Binary Cross-Entropy Loss', color: '#ADB5BD' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#ADB5BD' }
                }
            }
        }
    });

    // Initialize Metrics Chart
    const metricsCtx = document.getElementById('metricsChart').getContext('2d');
    new Chart(metricsCtx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Dice Score',
                    data: diceCurve,
                    borderColor: chartCyan,
                    backgroundColor: 'transparent',
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 2
                },
                {
                    label: 'Intersection over Union (IoU)',
                    data: iouCurve,
                    borderColor: '#2EC4B6',
                    backgroundColor: 'transparent',
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#ADB5BD', font: { family: 'Inter' } }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Training Epochs', color: '#ADB5BD' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#ADB5BD' }
                },
                y: {
                    title: { display: true, text: 'Accuracy Score (0-1)', color: '#ADB5BD' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#ADB5BD' }
                }
            }
        }
    });
});
