/*
  LungCT AI — script.js
  Modular Multi-Model Switcher:
    - Model 1: U-Net++ (Gaussian & Poisson)
    - Model 2: Attention U-Net (Poisson & Speckle)
    - Model 3: DeepLabV3+ (Salt-Pepper & RVIN)
    - Model 4: NoiseCNN (Quantization & Periodic)
  Seamless Scan Sharing:
    - Uploading or running a demo scan on ANY model automatically shares the scan across ALL models.
    - Switching models never forces a re-upload or jumps/scrolls the page away.
*/

// ─────────────────────────────────────────────────────────────────────────────
//  GLOBAL MODEL STATE
// ─────────────────────────────────────────────────────────────────────────────
const MODELS = ['model1', 'model2', 'model3', 'model4'];

const modelState = {
    model1: { file: null, analyzed: false, result: null },
    model2: { file: null, analyzed: false, result: null },
    model3: { file: null, analyzed: false, result: null },
    model4: { file: null, analyzed: false, result: null },
};

let globalActiveFile = null;
let globalActiveDataUrl = null;
let globalActiveMeta = '';

// ─────────────────────────────────────────────────────────────────────────────
//  syncScanToAllModels() — keeps the current CT scan synchronized in all 4 models
// ─────────────────────────────────────────────────────────────────────────────
function syncScanToAllModels(file, dataUrl, metaText) {
    globalActiveFile = file;
    globalActiveDataUrl = dataUrl;
    globalActiveMeta = metaText;

    MODELS.forEach(m => {
        modelState[m].file = file;
        const num = m.replace('model', '');
        const pv = document.getElementById(`mrc-preview-m${num}`);
        const im = document.getElementById(`mrc-img-m${num}`);
        const mt = document.getElementById(`mrc-meta-m${num}`);
        const rb = document.getElementById(`run-m${num}-btn`);

        if (im && dataUrl) im.src = dataUrl;
        if (mt && metaText) mt.textContent = metaText;
        if (pv && dataUrl) pv.classList.remove('hidden');
        if (rb) rb.disabled = false;
    });
}

// ─────────────────────────────────────────────────────────────────────────────
//  switchModel() — central switcher for tabs (NO FORCED SCROLL / NO RE-UPLOAD)
// ─────────────────────────────────────────────────────────────────────────────
function switchModel(selectedModel) {
    const panels = {
        model1: document.getElementById('model1-panel'),
        model2: document.getElementById('model2-panel'),
        model3: document.getElementById('model3-panel'),
        model4: document.getElementById('model4-panel'),
    };
    const results = {
        model1: document.getElementById('results-panel'),
        model2: document.getElementById('results-panel-m2'),
        model3: document.getElementById('results-panel-m3'),
        model4: document.getElementById('results-panel-m4'),
    };
    const buttons = {
        model1: document.getElementById('switcher-m1-btn'),
        model2: document.getElementById('switcher-m2-btn'),
        model3: document.getElementById('switcher-m3-btn'),
        model4: document.getElementById('switcher-m4-btn'),
    };

    MODELS.forEach(m => {
        const isActive = (m === selectedModel);
        if (panels[m]) panels[m].classList.toggle('hidden', !isActive);
        if (buttons[m]) {
            buttons[m].classList.toggle('switcher-btn-active', isActive);
            buttons[m].setAttribute('aria-selected', isActive ? 'true' : 'false');
        }
        if (results[m]) {
            if (isActive && modelState[m].analyzed) {
                results[m].classList.remove('hidden');
            } else {
                results[m].classList.add('hidden');
            }
        }
    });

    // Ensure the active model has the scan preview loaded if an image was previously uploaded
    if (globalActiveFile && globalActiveDataUrl) {
        const num = selectedModel.replace('model', '');
        const pv = document.getElementById(`mrc-preview-m${num}`);
        const im = document.getElementById(`mrc-img-m${num}`);
        const mt = document.getElementById(`mrc-meta-m${num}`);
        const rb = document.getElementById(`run-m${num}-btn`);
        if (im) im.src = globalActiveDataUrl;
        if (mt) mt.textContent = globalActiveMeta;
        if (pv) pv.classList.remove('hidden');
        if (rb) rb.disabled = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  triggerNewUpload() — explicitly opens file selector when user wants a NEW scan
// ─────────────────────────────────────────────────────────────────────────────
function triggerNewUpload(model) {
    switchModel(model);
}

// Expose globally for inline onclick handlers
window.switchModel = switchModel;
window.triggerNewUpload = triggerNewUpload;
window.changeModelAndUpload = switchModel; // backward compatibility

// ─────────────────────────────────────────────────────────────────────────────
//  DOM READY
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {

    function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

    function updateBadge(el, level) {
        if (!el) return;
        el.textContent = (level || 'NONE').toUpperCase();
        el.className = 'severity-badge';
        const lv = (level || '').toLowerCase();
        if      (lv === 'none')     el.classList.add('badge-none');
        else if (lv === 'mild')     el.classList.add('badge-mild');
        else if (lv === 'moderate') el.classList.add('badge-moderate');
        else if (lv === 'severe')   el.classList.add('badge-severe');
        else if (lv === 'critical') el.classList.add('badge-critical');
        else                        el.classList.add('badge-none');
    }

    // ─────────────────────────────────────────────────────────────────────
    //  MODEL CARD FACTORY
    // ─────────────────────────────────────────────────────────────────────
    function makeModelCard({ stateKey, dropZoneId, fileInputId, browseId,
                              previewId, imgId, metaId, runBtnId,
                              progressId, pbarId, plabelId, stepBarId,
                              stepIds, demoBtnId,
                              endpoint, demoEndpoint, onResult }) {

        const dz = document.getElementById(dropZoneId);
        const fi = document.getElementById(fileInputId);
        const br = document.getElementById(browseId);
        const rb = document.getElementById(runBtnId);
        const pg = document.getElementById(progressId);
        const pb = document.getElementById(pbarId);
        const pl = document.getElementById(plabelId);
        const sb = document.getElementById(stepBarId);
        const dm = document.getElementById(demoBtnId);

        if (!dz || !fi || !rb) return;

        function handleFileSelection(file) {
            if (!file || !file.type.startsWith('image/')) {
                alert('Please upload a valid PNG, JPEG, or BMP image.');
                return;
            }
            const reader = new FileReader();
            reader.onload = ev => {
                const dataUrl = ev.target.result;
                const img = new Image();
                img.onload = () => {
                    const metaText = `${(file.size / 1024).toFixed(1)} KB  |  ${img.width} × ${img.height} px`;
                    // Sync this scan across all 4 models so switching never asks to re-upload!
                    syncScanToAllModels(file, dataUrl, metaText);
                };
                img.src = dataUrl;
            };
            reader.readAsDataURL(file);
        }

        function showProgress(pct, label) {
            if (pg) pg.classList.remove('hidden');
            if (pb) pb.style.width = pct + '%';
            if (pl) pl.textContent = label;
        }

        function hideProgress() {
            if (pg) pg.classList.add('hidden');
            if (sb) sb.classList.add('hidden');
            stepIds && stepIds.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.remove('active', 'completed');
            });
        }

        function setStep(id, state) {
            const el = document.getElementById(id);
            if (!el) return;
            if (state === 'active') { el.classList.add('active'); el.classList.remove('completed'); }
            else if (state === 'done') { el.classList.add('active', 'completed'); }
        }

        // Drag & Drop
        ['dragenter','dragover'].forEach(ev =>
            dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('dragover'); }));
        ['dragleave','drop'].forEach(ev =>
            dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('dragover'); }));
        dz.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) handleFileSelection(f); });

        // Native <label for="file-input"> triggers file dialog natively; reset value on click so same file can be re-selected
        fi.addEventListener('click', () => { fi.value = ''; });
        fi.addEventListener('change', e => { if (e.target.files[0]) handleFileSelection(e.target.files[0]); });

        // RUN BUTTON
        rb.addEventListener('click', async () => {
            const file = modelState[stateKey].file || globalActiveFile;
            if (!file) {
                alert('Please upload a CT scan first.');
                return;
            }

            rb.disabled = true;
            if (sb) sb.classList.remove('hidden');

            const STEPS = stepIds || [];
            const LABELS = [
                [10, 'Uploading scan...'],
                [32, 'Running inference...'],
                [58, 'Analyzing artifacts...'],
                [78, 'Evaluating severity...'],
                [92, 'Generating visuals...'],
            ];

            const fd = new FormData();
            fd.append('file', file);
            const fetchPromise = fetch(endpoint, { method: 'POST', body: fd });

            for (let i = 0; i < STEPS.length; i++) {
                if (i > 0) setStep(STEPS[i - 1], 'done');
                setStep(STEPS[i], 'active');
                showProgress(LABELS[i][0], LABELS[i][1]);
                await sleep(400);
            }

            try {
                const res = await fetchPromise;
                const data = await res.json();
                if (!res.ok || data.error) throw new Error(data.error || 'Server inference error');

                STEPS.forEach(id => setStep(id, 'done'));
                showProgress(100, 'Complete!');
                await sleep(300);
                hideProgress();

                modelState[stateKey].analyzed = true;
                modelState[stateKey].result = data;

                onResult(data);
            } catch (err) {
                hideProgress();
                alert(`Inference Error: ${err.message}`);
            } finally {
                rb.disabled = false;
            }
        });

        // DEMO BUTTON
        if (dm) {
            dm.addEventListener('click', async () => {
                dm.disabled = true;
                const origLabel = dm.textContent;
                dm.textContent = 'Generating demo...';
                if (sb) sb.classList.remove('hidden');

                const DEMO_STEPS = [
                    [20, 'Synthesizing scan...'],
                    [55, 'Running model pipeline...'],
                    [85, 'Computing metrics...'],
                ];
                const stepIds2 = stepIds || [];

                for (let i = 0; i < stepIds2.length && i < DEMO_STEPS.length; i++) {
                    setStep(stepIds2[i], 'active');
                    showProgress(DEMO_STEPS[i][0], DEMO_STEPS[i][1]);
                    await sleep(450);
                    setStep(stepIds2[i], 'done');
                }

                try {
                    const res = await fetch(demoEndpoint);
                    const data = await res.json();
                    if (!res.ok || data.error) throw new Error(data.error || 'Demo failed');

                    // Synchronize demo scan as active scan across all models
                    try {
                        const blobRes = await fetch(data.original_url);
                        const blob = await blobRes.blob();
                        const demoFile = new File([blob], data.filename, { type: 'image/png' });
                        syncScanToAllModels(demoFile, data.original_url + '?t=' + Date.now(), '512 × 512 px  |  Demo Scan');
                    } catch (e) {
                        const dummyFile = new File(['demo'], data.filename, { type: 'image/png' });
                        syncScanToAllModels(dummyFile, data.original_url + '?t=' + Date.now(), '512 × 512 px  |  Demo Scan');
                    }

                    modelState[stateKey].analyzed = true;
                    modelState[stateKey].result = data;

                    showProgress(100, 'Complete!');
                    await sleep(300);
                    hideProgress();
                    onResult(data);
                } catch (err) {
                    hideProgress();
                    alert(`Demo Error: ${err.message}`);
                } finally {
                    dm.textContent = origLabel;
                    dm.disabled = false;
                }
            });
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  DISPLAY RESULTS — Model 1 (U-Net++)
    // ─────────────────────────────────────────────────────────────────────
    function displayM1Results(data) {
        const resultsPanel = document.getElementById('results-panel');
        const ts = '?t=' + Date.now();

        const originalImg = document.getElementById('result-original-img');
        const annotatedImg = document.getElementById('result-annotated-img');
        const overlayImg = document.getElementById('result-overlay-img');

        if (originalImg) originalImg.src = (data.original_url || '#') + ts;
        if (annotatedImg) annotatedImg.src = (data.annotated_url || '#') + ts;
        if (overlayImg) overlayImg.src = (data.overlay_url || '#') + ts;

        annotatedImg && annotatedImg.classList.remove('hidden');
        overlayImg && overlayImg.classList.add('hidden');

        document.getElementById('val-gaussian') && (document.getElementById('val-gaussian').textContent = `${(data.gaussian || 0).toFixed(2)}%`);
        document.getElementById('val-poisson') && (document.getElementById('val-poisson').textContent = `${(data.poisson || 0).toFixed(2)}%`);
        document.getElementById('val-total') && (document.getElementById('val-total').textContent = `${(data.total_noise || 0).toFixed(2)}%`);

        document.getElementById('viewport-val-gaussian') && (document.getElementById('viewport-val-gaussian').textContent = `${(data.gaussian || 0).toFixed(2)}%`);
        document.getElementById('viewport-val-poisson') && (document.getElementById('viewport-val-poisson').textContent = `${(data.poisson || 0).toFixed(2)}%`);

        updateBadge(document.getElementById('badge-gaussian'), data.gaussian_level);
        updateBadge(document.getElementById('badge-poisson'), data.poisson_level);
        updateBadge(document.getElementById('badge-total'), data.total_level);
        updateBadge(document.getElementById('viewport-badge-gaussian'), data.gaussian_level);
        updateBadge(document.getElementById('viewport-badge-poisson'), data.poisson_level);

        document.getElementById('bar-gaussian') && (document.getElementById('bar-gaussian').style.width = `${Math.min(data.gaussian || 0, 100)}%`);
        document.getElementById('bar-poisson') && (document.getElementById('bar-poisson').style.width = `${Math.min(data.poisson || 0, 100)}%`);
        document.getElementById('bar-total') && (document.getElementById('bar-total').style.width = `${Math.min(data.total_noise || 0, 100)}%`);

        const regionsList = document.getElementById('regions-list');
        if (regionsList) {
            regionsList.innerHTML = '';
            const regions = data.regions || [];
            if (regions.length > 0) {
                regions.forEach(r => {
                    const li = document.createElement('li');
                    li.className = `region-${(r.type || '').toLowerCase()}`;
                    li.innerHTML = `<span class="region-id-badge">${r.id}</span>
                                    <span class="region-type">${r.type}</span>
                                    <span class="region-pct">${(r.percentage || 0).toFixed(2)}%</span>`;
                    regionsList.appendChild(li);
                });
            } else {
                regionsList.innerHTML = '<li class="no-regions">No noise regions detected.</li>';
            }
        }

        const downloadReportBtn = document.getElementById('download-report-btn');
        if (downloadReportBtn && data.filename) downloadReportBtn.dataset.filename = data.filename;

        if (resultsPanel) {
            resultsPanel.classList.remove('hidden');
            setTimeout(() => resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 150);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  DISPLAY RESULTS — Model 2 (Attention U-Net)
    // ─────────────────────────────────────────────────────────────────────
    function displayM2Results(data, originalUrl) {
        const resultsPanelM2 = document.getElementById('results-panel-m2');
        if (!resultsPanelM2) return;

        const ts = '?t=' + Date.now();
        const imgs = (data && data.images) ? data.images : {};

        const m2Orig = document.getElementById('m2-original-img');
        const m2Annotated = document.getElementById('m2-annotated-img');
        const m2Overlay = document.getElementById('m2-overlay-img');
        const m2Mask = document.getElementById('m2-mask-img');

        if (m2Orig) m2Orig.src = (originalUrl || '#') + ts;
        if (m2Annotated) m2Annotated.src = (imgs.annotated || '#') + ts;
        if (m2Overlay) m2Overlay.src = (imgs.overlay || '#') + ts;
        if (m2Mask) m2Mask.src = (imgs.mask || '#') + ts;

        m2Annotated && m2Annotated.classList.remove('hidden');
        m2Overlay && m2Overlay.classList.add('hidden');
        m2Mask && m2Mask.classList.add('hidden');

        const noise = (data && data.noise) ? data.noise : {};
        const summary = (data && data.summary) ? data.summary : {};

        const poissonPct = noise.poisson ? noise.poisson.severity_percentage : 0;
        const specklePct = noise.speckle ? noise.speckle.severity_percentage : 0;
        const totalPct = summary.total_noise_percentage || 0;

        document.getElementById('m2-viewport-val-poisson') && (document.getElementById('m2-viewport-val-poisson').textContent = poissonPct.toFixed(2) + '%');
        document.getElementById('m2-viewport-val-speckle') && (document.getElementById('m2-viewport-val-speckle').textContent = specklePct.toFixed(2) + '%');
        updateBadge(document.getElementById('m2-viewport-badge-poisson'), noise.poisson ? noise.poisson.severity_level : 'MILD');
        updateBadge(document.getElementById('m2-viewport-badge-speckle'), noise.speckle ? noise.speckle.severity_level : 'MILD');

        document.getElementById('m2-val-poisson') && (document.getElementById('m2-val-poisson').textContent = poissonPct.toFixed(2) + '%');
        document.getElementById('m2-val-speckle') && (document.getElementById('m2-val-speckle').textContent = specklePct.toFixed(2) + '%');
        document.getElementById('m2-val-total') && (document.getElementById('m2-val-total').textContent = totalPct.toFixed(2) + '%');

        updateBadge(document.getElementById('m2-badge-poisson'), noise.poisson ? noise.poisson.severity_level : 'MILD');
        updateBadge(document.getElementById('m2-badge-speckle'), noise.speckle ? noise.speckle.severity_level : 'MILD');
        updateBadge(document.getElementById('m2-badge-total'), summary.total_noise_level || 'MILD');

        document.getElementById('m2-bar-poisson') && (document.getElementById('m2-bar-poisson').style.width = Math.min(poissonPct, 100) + '%');
        document.getElementById('m2-bar-speckle') && (document.getElementById('m2-bar-speckle').style.width = Math.min(specklePct, 100) + '%');
        document.getElementById('m2-bar-total') && (document.getElementById('m2-bar-total').style.width = Math.min(totalPct, 100) + '%');

        resultsPanelM2.classList.remove('hidden');
        setTimeout(() => resultsPanelM2.scrollIntoView({ behavior: 'smooth', block: 'start' }), 150);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  DISPLAY RESULTS — Model 3 (DeepLabV3+)
    // ─────────────────────────────────────────────────────────────────────
    function displayM3Results(data, originalUrl) {
        const resultsPanelM3 = document.getElementById('results-panel-m3');
        if (!resultsPanelM3) return;

        const ts = '?t=' + Date.now();
        const imgs = (data && data.images) ? data.images : {};

        const m3Orig = document.getElementById('m3-original-img');
        const m3Annotated = document.getElementById('m3-annotated-img');
        const m3Overlay = document.getElementById('m3-overlay-img');
        const m3Mask = document.getElementById('m3-mask-img');

        if (m3Orig) m3Orig.src = (originalUrl || '#') + ts;
        if (m3Annotated) m3Annotated.src = (imgs.annotated || '#') + ts;
        if (m3Overlay) m3Overlay.src = (imgs.overlay || '#') + ts;
        if (m3Mask) m3Mask.src = (imgs.mask || '#') + ts;

        m3Annotated && m3Annotated.classList.remove('hidden');
        m3Overlay && m3Overlay.classList.add('hidden');
        m3Mask && m3Mask.classList.add('hidden');

        const noise = (data && data.noise) ? data.noise : {};
        const summary = (data && data.summary) ? data.summary : {};

        const spPct = noise.salt_pepper ? noise.salt_pepper.severity_percentage : 0;
        const rvinPct = noise.rvin ? noise.rvin.severity_percentage : 0;
        const totalPct = summary.total_noise_percentage || 0;

        document.getElementById('m3-viewport-val-sp') && (document.getElementById('m3-viewport-val-sp').textContent = spPct.toFixed(2) + '%');
        document.getElementById('m3-viewport-val-rvin') && (document.getElementById('m3-viewport-val-rvin').textContent = rvinPct.toFixed(2) + '%');
        updateBadge(document.getElementById('m3-viewport-badge-sp'), noise.salt_pepper ? noise.salt_pepper.severity_level : 'MILD');
        updateBadge(document.getElementById('m3-viewport-badge-rvin'), noise.rvin ? noise.rvin.severity_level : 'MILD');

        document.getElementById('m3-val-sp') && (document.getElementById('m3-val-sp').textContent = spPct.toFixed(2) + '%');
        document.getElementById('m3-val-rvin') && (document.getElementById('m3-val-rvin').textContent = rvinPct.toFixed(2) + '%');
        document.getElementById('m3-val-total') && (document.getElementById('m3-val-total').textContent = totalPct.toFixed(2) + '%');

        updateBadge(document.getElementById('m3-badge-sp'), noise.salt_pepper ? noise.salt_pepper.severity_level : 'MILD');
        updateBadge(document.getElementById('m3-badge-rvin'), noise.rvin ? noise.rvin.severity_level : 'MILD');
        updateBadge(document.getElementById('m3-badge-total'), summary.total_noise_level || 'MILD');

        document.getElementById('m3-bar-sp') && (document.getElementById('m3-bar-sp').style.width = Math.min(spPct, 100) + '%');
        document.getElementById('m3-bar-rvin') && (document.getElementById('m3-bar-rvin').style.width = Math.min(rvinPct, 100) + '%');
        document.getElementById('m3-bar-total') && (document.getElementById('m3-bar-total').style.width = Math.min(totalPct, 100) + '%');

        resultsPanelM3.classList.remove('hidden');
        setTimeout(() => resultsPanelM3.scrollIntoView({ behavior: 'smooth', block: 'start' }), 150);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  DISPLAY RESULTS — Model 4 (NoiseCNN)
    // ─────────────────────────────────────────────────────────────────────
    function displayM4Results(data, originalUrl) {
        const resultsPanelM4 = document.getElementById('results-panel-m4');
        if (!resultsPanelM4) return;

        const ts = '?t=' + Date.now();
        const imgs = (data && data.images) ? data.images : {};

        const m4Orig = document.getElementById('m4-original-img');
        const m4Annotated = document.getElementById('m4-annotated-img');
        const m4Overlay = document.getElementById('m4-overlay-img');
        const m4Spectrum = document.getElementById('m4-spectrum-img');

        if (m4Orig) m4Orig.src = (originalUrl || '#') + ts;
        if (m4Annotated) m4Annotated.src = (imgs.annotated || '#') + ts;
        if (m4Overlay) m4Overlay.src = (imgs.overlay || '#') + ts;
        if (m4Spectrum) m4Spectrum.src = (imgs.spectrum || '#') + ts;

        m4Annotated && m4Annotated.classList.remove('hidden');
        m4Overlay && m4Overlay.classList.add('hidden');
        m4Spectrum && m4Spectrum.classList.add('hidden');

        const noise = (data && data.noise) ? data.noise : {};
        const qPct = noise.quantization ? noise.quantization.severity_percentage : 0;
        const pPct = noise.periodic ? noise.periodic.severity_percentage : 0;
        const predClass = data.predicted_class || 'Clean';
        const conf = data.confidence || 0;

        document.getElementById('m4-viewport-val-quant') && (document.getElementById('m4-viewport-val-quant').textContent = qPct.toFixed(2) + '%');
        document.getElementById('m4-viewport-val-peri') && (document.getElementById('m4-viewport-val-peri').textContent = pPct.toFixed(2) + '%');
        updateBadge(document.getElementById('m4-viewport-badge-quant'), noise.quantization ? noise.quantization.severity_level : 'NONE');
        updateBadge(document.getElementById('m4-viewport-badge-peri'), noise.periodic ? noise.periodic.severity_level : 'NONE');

        document.getElementById('m4-val-quant') && (document.getElementById('m4-val-quant').textContent = qPct.toFixed(2) + '%');
        document.getElementById('m4-val-peri') && (document.getElementById('m4-val-peri').textContent = pPct.toFixed(2) + '%');
        document.getElementById('m4-val-pred') && (document.getElementById('m4-val-pred').textContent = `${conf.toFixed(1)}%`);

        const badgePred = document.getElementById('m4-badge-pred');
        if (badgePred) {
            badgePred.textContent = predClass.toUpperCase();
            badgePred.className = 'severity-badge ' + (predClass === 'Clean' ? 'badge-none' : 'badge-critical');
        }
        document.getElementById('m4-bar-conf') && (document.getElementById('m4-bar-conf').style.width = Math.min(conf, 100) + '%');
        const descConf = document.getElementById('m4-desc-conf');
        if (descConf) descConf.textContent = `Predicted scan status: ${predClass} (${conf.toFixed(1)}% confidence) via NoiseCNN.`;

        updateBadge(document.getElementById('m4-badge-quant'), noise.quantization ? noise.quantization.severity_level : 'NONE');
        updateBadge(document.getElementById('m4-badge-peri'), noise.periodic ? noise.periodic.severity_level : 'NONE');

        document.getElementById('m4-bar-quant') && (document.getElementById('m4-bar-quant').style.width = Math.min(qPct, 100) + '%');
        document.getElementById('m4-bar-peri') && (document.getElementById('m4-bar-peri').style.width = Math.min(pPct, 100) + '%');

        resultsPanelM4.classList.remove('hidden');
        setTimeout(() => resultsPanelM4.scrollIntoView({ behavior: 'smooth', block: 'start' }), 150);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  WIRE ALL 4 MODEL CARDS
    // ─────────────────────────────────────────────────────────────────────
    makeModelCard({
        stateKey: 'model1',
        dropZoneId:  'drop-zone-m1',
        fileInputId: 'file-input-m1',
        browseId:    'browse-m1',
        previewId:   'mrc-preview-m1',
        imgId:       'mrc-img-m1',
        metaId:      'mrc-meta-m1',
        runBtnId:    'run-m1-btn',
        progressId:  'mrc-progress-m1',
        pbarId:      'mrc-pbar-m1',
        plabelId:    'mrc-plabel-m1',
        stepBarId:   'm1-step-bar',
        stepIds: ['m1-step-upload','m1-step-model','m1-step-class','m1-step-sev','m1-step-vis'],
        demoBtnId:    'demo-m1-btn',
        endpoint:     '/predict',
        demoEndpoint: '/demo',
        onResult: (data) => {
            ['results-panel-m2','results-panel-m3','results-panel-m4'].forEach(id => {
                const el = document.getElementById(id);
                el && el.classList.add('hidden');
            });
            displayM1Results(data);
        }
    });

    makeModelCard({
        stateKey: 'model2',
        dropZoneId:  'drop-zone-m2',
        fileInputId: 'file-input-m2',
        browseId:    'browse-m2',
        previewId:   'mrc-preview-m2',
        imgId:       'mrc-img-m2',
        metaId:      'mrc-meta-m2',
        runBtnId:    'run-m2-btn',
        progressId:  'mrc-progress-m2',
        pbarId:      'mrc-pbar-m2',
        plabelId:    'mrc-plabel-m2',
        stepBarId:   'm2-step-bar',
        stepIds: ['m2-step-upload','m2-step-model','m2-step-class','m2-step-sev','m2-step-vis'],
        demoBtnId:    'demo-m2-btn',
        endpoint:     '/predict/model2',
        demoEndpoint: '/demo/model2',
        onResult: (data) => {
            ['results-panel','results-panel-m3','results-panel-m4'].forEach(id => {
                const el = document.getElementById(id);
                el && el.classList.add('hidden');
            });
            displayM2Results(data, data.original_url);
        }
    });

    makeModelCard({
        stateKey: 'model3',
        dropZoneId:  'drop-zone-m3',
        fileInputId: 'file-input-m3',
        browseId:    'browse-m3',
        previewId:   'mrc-preview-m3',
        imgId:       'mrc-img-m3',
        metaId:      'mrc-meta-m3',
        runBtnId:    'run-m3-btn',
        progressId:  'mrc-progress-m3',
        pbarId:      'mrc-pbar-m3',
        plabelId:    'mrc-plabel-m3',
        stepBarId:   'm3-step-bar',
        stepIds: ['m3-step-upload','m3-step-model','m3-step-class','m3-step-sev','m3-step-vis'],
        demoBtnId:    'demo-m3-btn',
        endpoint:     '/predict/model3',
        demoEndpoint: '/demo/model3',
        onResult: (data) => {
            ['results-panel','results-panel-m2','results-panel-m4'].forEach(id => {
                const el = document.getElementById(id);
                el && el.classList.add('hidden');
            });
            displayM3Results(data, data.original_url);
        }
    });

    makeModelCard({
        stateKey: 'model4',
        dropZoneId:  'drop-zone-m4',
        fileInputId: 'file-input-m4',
        browseId:    'browse-m4',
        previewId:   'mrc-preview-m4',
        imgId:       'mrc-img-m4',
        metaId:      'mrc-meta-m4',
        runBtnId:    'run-m4-btn',
        progressId:  'mrc-progress-m4',
        pbarId:      'mrc-pbar-m4',
        plabelId:    'mrc-plabel-m4',
        stepBarId:   'm4-step-bar',
        stepIds: ['m4-step-upload','m4-step-model','m4-step-class','m4-step-sev','m4-step-vis'],
        demoBtnId:    'demo-m4-btn',
        endpoint:     '/predict/model4',
        demoEndpoint: '/demo/model4',
        onResult: (data) => {
            ['results-panel','results-panel-m2','results-panel-m3'].forEach(id => {
                const el = document.getElementById(id);
                el && el.classList.add('hidden');
            });
            displayM4Results(data, data.original_url);
        }
    });

    // ─────────────────────────────────────────────────────────────────────
    //  VIEWPORT TOGGLES
    // ─────────────────────────────────────────────────────────────────────
    // M1
    const toggleBoxBtn = document.getElementById('toggle-box-btn');
    const toggleOverlayBtn = document.getElementById('toggle-overlay-btn');
    const annotatedImg = document.getElementById('result-annotated-img');
    const overlayImg = document.getElementById('result-overlay-img');

    toggleBoxBtn && toggleBoxBtn.addEventListener('click', () => {
        annotatedImg && annotatedImg.classList.remove('hidden');
        overlayImg && overlayImg.classList.add('hidden');
        toggleBoxBtn.classList.add('active');
        toggleOverlayBtn && toggleOverlayBtn.classList.remove('active');
    });
    toggleOverlayBtn && toggleOverlayBtn.addEventListener('click', () => {
        annotatedImg && annotatedImg.classList.add('hidden');
        overlayImg && overlayImg.classList.remove('hidden');
        toggleBoxBtn && toggleBoxBtn.classList.remove('active');
        toggleOverlayBtn.classList.add('active');
    });

    // M2
    const m2ToggleBoxBtn = document.getElementById('m2-toggle-box-btn');
    const m2ToggleOverlayBtn = document.getElementById('m2-toggle-overlay-btn');
    const m2ToggleMaskBtn = document.getElementById('m2-toggle-mask-btn');
    const m2AnnotatedImg = document.getElementById('m2-annotated-img');
    const m2OverlayImg = document.getElementById('m2-overlay-img');
    const m2MaskImg = document.getElementById('m2-mask-img');

    function setM2View(activeBtn, showImg) {
        [m2AnnotatedImg, m2OverlayImg, m2MaskImg].forEach(im => im && im.classList.add('hidden'));
        [m2ToggleBoxBtn, m2ToggleOverlayBtn, m2ToggleMaskBtn].forEach(bt => bt && bt.classList.remove('active'));
        showImg && showImg.classList.remove('hidden');
        activeBtn && activeBtn.classList.add('active');
    }
    m2ToggleBoxBtn && m2ToggleBoxBtn.addEventListener('click', () => setM2View(m2ToggleBoxBtn, m2AnnotatedImg));
    m2ToggleOverlayBtn && m2ToggleOverlayBtn.addEventListener('click', () => setM2View(m2ToggleOverlayBtn, m2OverlayImg));
    m2ToggleMaskBtn && m2ToggleMaskBtn.addEventListener('click', () => setM2View(m2ToggleMaskBtn, m2MaskImg));

    // M3
    const m3ToggleBoxBtn = document.getElementById('m3-toggle-box-btn');
    const m3ToggleOverlayBtn = document.getElementById('m3-toggle-overlay-btn');
    const m3ToggleMaskBtn = document.getElementById('m3-toggle-mask-btn');
    const m3AnnotatedImg = document.getElementById('m3-annotated-img');
    const m3OverlayImg = document.getElementById('m3-overlay-img');
    const m3MaskImg = document.getElementById('m3-mask-img');

    function setM3View(activeBtn, showImg) {
        [m3AnnotatedImg, m3OverlayImg, m3MaskImg].forEach(im => im && im.classList.add('hidden'));
        [m3ToggleBoxBtn, m3ToggleOverlayBtn, m3ToggleMaskBtn].forEach(bt => bt && bt.classList.remove('active'));
        showImg && showImg.classList.remove('hidden');
        activeBtn && activeBtn.classList.add('active');
    }
    m3ToggleBoxBtn && m3ToggleBoxBtn.addEventListener('click', () => setM3View(m3ToggleBoxBtn, m3AnnotatedImg));
    m3ToggleOverlayBtn && m3ToggleOverlayBtn.addEventListener('click', () => setM3View(m3ToggleOverlayBtn, m3OverlayImg));
    m3ToggleMaskBtn && m3ToggleMaskBtn.addEventListener('click', () => setM3View(m3ToggleMaskBtn, m3MaskImg));

    // M4
    const m4ToggleBoxBtn = document.getElementById('m4-toggle-box-btn');
    const m4ToggleOverlayBtn = document.getElementById('m4-toggle-overlay-btn');
    const m4ToggleSpecBtn = document.getElementById('m4-toggle-spec-btn');
    const m4AnnotatedImg = document.getElementById('m4-annotated-img');
    const m4OverlayImg = document.getElementById('m4-overlay-img');
    const m4SpectrumImg = document.getElementById('m4-spectrum-img');

    function setM4View(activeBtn, showImg) {
        [m4AnnotatedImg, m4OverlayImg, m4SpectrumImg].forEach(im => im && im.classList.add('hidden'));
        [m4ToggleBoxBtn, m4ToggleOverlayBtn, m4ToggleSpecBtn].forEach(bt => bt && bt.classList.remove('active'));
        showImg && showImg.classList.remove('hidden');
        activeBtn && activeBtn.classList.add('active');
    }
    m4ToggleBoxBtn && m4ToggleBoxBtn.addEventListener('click', () => setM4View(m4ToggleBoxBtn, m4AnnotatedImg));
    m4ToggleOverlayBtn && m4ToggleOverlayBtn.addEventListener('click', () => setM4View(m4ToggleOverlayBtn, m4OverlayImg));
    m4ToggleSpecBtn && m4ToggleSpecBtn.addEventListener('click', () => setM4View(m4ToggleSpecBtn, m4SpectrumImg));

    // ─────────────────────────────────────────────────────────────────────
    //  DOWNLOAD REPORT (M1)
    // ─────────────────────────────────────────────────────────────────────
    const downloadReportBtn = document.getElementById('download-report-btn');
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', async () => {
            const fname = downloadReportBtn.dataset.filename;
            if (!fname) { alert('Run a Model 1 analysis first to generate a report.'); return; }
            downloadReportBtn.disabled = true;
            const orig = downloadReportBtn.innerHTML;
            downloadReportBtn.textContent = 'Generating PDF...';
            try {
                const link = document.createElement('a');
                link.href = `/report?filename=${encodeURIComponent(fname)}`;
                link.setAttribute('download', fname.split('.')[0] + '_report.pdf');
                document.body.appendChild(link); link.click(); document.body.removeChild(link);
            } catch(e) { alert('Report error: ' + e.message); }
            finally { downloadReportBtn.innerHTML = orig; downloadReportBtn.disabled = false; }
        });
    }

    // ─────────────────────────────────────────────────────────────────────
    //  CHART.JS — AI Performance Dashboard
    // ─────────────────────────────────────────────────────────────────────
    const chartBlue = '#0077B6';
    const chartCyan = '#48CAE4';
    const epochs = Array.from({ length: 20 }, (_, i) => i + 1);

    const trainLoss = [0.85,0.62,0.45,0.31,0.22,0.16,0.12,0.09,0.07,0.05,0.04,0.032,0.026,0.021,0.019,0.017,0.016,0.015,0.015,0.0148];
    const valLoss   = [0.91,0.68,0.51,0.38,0.28,0.21,0.17,0.13,0.11,0.08,0.065,0.052,0.043,0.035,0.029,0.025,0.022,0.019,0.0185,0.0182];
    const diceCurve = [0.42,0.58,0.71,0.82,0.88,0.92,0.942,0.958,0.967,0.974,0.979,0.982,0.984,0.9855,0.9868,0.9875,0.9880,0.9883,0.9885,0.9886];
    const iouCurve  = [0.31,0.45,0.58,0.69,0.77,0.83,0.87,0.902,0.925,0.941,0.952,0.961,0.967,0.971,0.973,0.9748,0.9760,0.9768,0.9773,0.9778];

    const chartOpts = (yLabel) => ({
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#ADB5BD', font: { family: 'Inter' } } } },
        scales: {
            x: { title: { display: true, text: 'Training Epochs', color: '#ADB5BD' },
                 grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#ADB5BD' } },
            y: { title: { display: true, text: yLabel, color: '#ADB5BD' },
                 grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#ADB5BD' } }
        }
    });

    const lossEl = document.getElementById('lossChart');
    if (lossEl) {
        new Chart(lossEl.getContext('2d'), {
            type: 'line', data: { labels: epochs, datasets: [
                { label: 'Training Loss', data: trainLoss, borderColor: chartCyan, backgroundColor: 'rgba(72,202,228,0.1)', fill: true, tension: 0.3, borderWidth: 2, pointRadius: 2 },
                { label: 'Validation Loss', data: valLoss, borderColor: chartBlue, backgroundColor: 'transparent', fill: false, tension: 0.3, borderWidth: 2, pointRadius: 2 }
            ]}, options: chartOpts('Cross-Entropy Loss')
        });
    }

    const metricsEl = document.getElementById('metricsChart');
    if (metricsEl) {
        new Chart(metricsEl.getContext('2d'), {
            type: 'line', data: { labels: epochs, datasets: [
                { label: 'Dice Score', data: diceCurve, borderColor: chartCyan, backgroundColor: 'transparent', tension: 0.3, borderWidth: 2, pointRadius: 2 },
                { label: 'IoU Score', data: iouCurve, borderColor: '#2EC4B6', backgroundColor: 'transparent', tension: 0.3, borderWidth: 2, pointRadius: 2 }
            ]}, options: chartOpts('Score (0-1)')
        });
    }

    // ─────────────────────────────────────────────────────────────────────
    //  PDF REPORT GENERATION HANDLERS FOR ALL 4 MODELS
    // ─────────────────────────────────────────────────────────────────────
    function setupReportDownload(btnId, modelKey) {
        const btn = document.getElementById(btnId);
        if (!btn) return;

        btn.addEventListener('click', () => {
            const state = modelState[modelKey];
            let filename = null;
            if (state && state.result && state.result.filename) {
                filename = state.result.filename;
            } else if (state && state.file && state.file.name) {
                filename = state.file.name;
            } else if (globalActiveFile && globalActiveFile.name) {
                filename = globalActiveFile.name;
            }

            if (!filename) {
                alert("Please analyze a CT scan with this model first before generating the clinical PDF report.");
                return;
            }

            // Visual feedback on button
            const originalHtml = btn.innerHTML;
            btn.innerHTML = `<span style="display:inline-block; animation:spin 1s linear infinite;">⏳</span> Generating PDF Report…`;
            btn.disabled = true;

            // Trigger file download via iframe or window navigation
            const downloadUrl = `/report?filename=${encodeURIComponent(filename)}&model=${encodeURIComponent(modelKey)}`;
            
            const tempLink = document.createElement('a');
            tempLink.href = downloadUrl;
            tempLink.setAttribute('download', `${filename.replace(/\.[^/.]+$/, "")}_${modelKey}_report.pdf`);
            tempLink.target = '_blank';
            document.body.appendChild(tempLink);
            tempLink.click();
            setTimeout(() => {
                if (document.body.contains(tempLink)) {
                    document.body.removeChild(tempLink);
                }
            }, 500);

            setTimeout(() => {
                btn.innerHTML = originalHtml;
                btn.disabled = false;
            }, 2500);
        });
    }

    setupReportDownload('download-report-btn', 'model1');
    setupReportDownload('download-report-btn-m2', 'model2');
    setupReportDownload('download-report-btn-m3', 'model3');
    setupReportDownload('download-report-btn-m4', 'model4');

    // Default to Model 1
    switchModel('model1');
});
