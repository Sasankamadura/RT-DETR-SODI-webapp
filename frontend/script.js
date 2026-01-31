const API_URL = "http://localhost:8000";

const dom = {
    modelList: document.getElementById('model-list'),
    currentModelName: document.getElementById('current-model-name'),
    uploadZone: document.getElementById('upload-zone'),
    fileInput: document.getElementById('file-input'),
    previewContainer: document.getElementById('preview-container'),
    originalPreview: document.getElementById('original-preview'),
    resultPreview: document.getElementById('result-preview'),
    loader: document.getElementById('processing-loader'),
    metricsGrid: document.getElementById('metrics-grid'),
    description: document.getElementById('model-description'),
    samplesGrid: document.getElementById('samples-grid'),
    downloadControls: document.getElementById('download-controls')
};

let currentModelId = null;
let modelsData = [];
let lastResult = null; // Store last result for downloading

// Init
async function init() {
    try {
        const res = await fetch(`${API_URL}/models`);
        modelsData = await res.json();
        renderModelList(modelsData);
        if (modelsData.length > 0) {
            selectModel(modelsData[0].id);
        }
        loadSampleImages();
    } catch (e) {
        console.error("Failed to fetch models", e);
        dom.modelList.innerHTML = `<p style="color:red; text-align:center">Error connecting to server</p>`;
    }
}

function renderModelList(models) {
    dom.modelList.innerHTML = models.map(model => `
        <div class="model-item" onclick="selectModel('${model.id}')" id="model-${model.id}">
            <span class="model-name">${model.name}</span>
        </div>
    `).join('');
}

function selectModel(id) {
    // Update UI active state
    document.querySelectorAll('.model-item').forEach(el => el.classList.remove('active'));
    document.getElementById(`model-${id}`)?.classList.add('active');

    currentModelId = id;
    const model = modelsData.find(m => m.id === id);

    // Update Header & Description
    dom.currentModelName.textContent = model.name;
    dom.description.innerHTML = `<p>${model.description}</p>`;

    // Update Summary Metrics
    dom.metricsGrid.innerHTML = Object.entries(model.summary_metrics).map(([key, value]) => `
        <div class="metric-card">
            <span class="label">${key.replace('_', ' ')}</span>
            <span class="value" style="color: ${value !== 'N/A' ? '#10b981' : '#9ca3af'}">${value}</span>
        </div>
    `).join('');

    // Populate Tabs
    renderDetailedMetrics(model);

    // Clear previous usage data
    clearPreview();
}

function clearPreview() {
    dom.fileInput.value = "";
    dom.previewContainer.classList.remove('visible');
    dom.previewContainer.classList.add('hidden');
    dom.originalPreview.src = "";
    dom.resultPreview.src = "";
    dom.uploadZone.style.borderColor = "";
    dom.downloadControls.classList.add('hidden');
    lastResult = null;
}

async function loadSampleImages() {
    try {
        const res = await fetch(`${API_URL}/samples`);
        const samples = await res.json();

        dom.samplesGrid.innerHTML = samples.map(s => `
            <div class="sample-item" onclick="selectSample('${s.url}')">
                <img src="${API_URL}${s.url}" alt="${s.filename}" loading="lazy">
            </div>
        `).join('');
    } catch (e) {
        console.error("Failed to load samples", e);
        dom.samplesGrid.innerHTML = '<p style="color:var(--text-sub)">Failed to load samples.</p>';
    }
}

async function selectSample(url) {
    if (!currentModelId) return alert("Please select a model first.");

    // Extract filename from URL (e.g. /samples-data/foo.jpg -> foo.jpg)
    // We decodeURIComponent just in case
    const filename = decodeURIComponent(url.split('/').pop());

    // Load image into preview
    dom.originalPreview.src = `${API_URL}${url}`;
    dom.resultPreview.src = "";
    dom.previewContainer.classList.remove('hidden');
    dom.previewContainer.classList.add('visible');

    // Direct server-side prediction
    const formData = new FormData();
    formData.append('model_id', currentModelId);
    formData.append('sample_filename', filename);

    dom.loader.classList.remove('hidden');

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Inference failed (${res.status}): ${errText}`);
        }

        const data = await res.json();

        if (!data.annotated_image) {
            throw new Error("Backend returned no image data");
        }

        lastResult = data;

        // Display result
        dom.resultPreview.src = `data:image/jpeg;base64,${data.annotated_image}`;
        dom.resultPreview.onerror = null; // Clear any previous handlers

        dom.downloadControls.classList.remove('hidden');

    } catch (e) {
        console.error(e);
        alert("Error during inference: " + e.message);
    } finally {
        dom.loader.classList.add('hidden');
    }
}
window.selectSample = selectSample;

function renderDetailedMetrics(model) {
    const full = model.full_metrics;

    // 1. Overview Tab
    let overviewHtml = '<table class="data-table"><tr><th>Metric</th><th>Value</th></tr>';

    // Params (calculated from profiling data if available)
    if (full.profiling && full.profiling.layer_analysis) {
        const totalParams = full.profiling.layer_analysis.total_params;
        overviewHtml += `
            <tr><td>Total Params</td><td>${(totalParams / 1e6).toFixed(2)} M</td></tr>
            <tr><td>Trainable</td><td>${(totalParams / 1e6).toFixed(2)} M</td></tr>
        `;
    } else if (full.evaluation) {
        // Fallback if profiling missing (though we prefer profiling)
        overviewHtml += `<tr><td>Params</td><td>N/A</td></tr>`;
    }

    if (full.fps && full.fps.fps_benchmark && full.fps.fps_benchmark['640x640']) {
        const bench = full.fps.fps_benchmark['640x640'];
        overviewHtml += `
            <tr><td>Latency Mean</td><td>${bench.latency_mean_ms.toFixed(2)} ms</td></tr>
            <tr><td>Latency Min</td><td>${bench.latency_min_ms.toFixed(2)} ms</td></tr>
            <tr><td>FPS Mean</td><td>${bench.fps_mean.toFixed(2)}</td></tr>
        `;
    }
    overviewHtml += '</table>';
    document.getElementById('tab-overview').innerHTML = overviewHtml;

    // 2. Class Metrics Tab
    let classHtml = '<table class="data-table"><tr><th>Class</th><th>AP 50</th><th>AP Small</th></tr>';
    if (full.detailed && full.detailed.per_class_metrics) {
        Object.values(full.detailed.per_class_metrics).forEach(m => {
            classHtml += `<tr>
                <td>${m.name}</td>
                <td>${(m.ap_50 * 100).toFixed(1)}%</td>
                <td>${(m.ap_small * 100).toFixed(1)}%</td>
            </tr>`;
        });
    } else {
        classHtml += '<tr><td colspan="3">No detailed class metrics available</td></tr>';
    }
    classHtml += '</table>';
    document.getElementById('tab-class-metrics').innerHTML = classHtml;

    // 3. Layer Analysis Tab
    let layerHtml = '';
    if (full.profiling && full.profiling.layer_analysis) {
        const la = full.profiling.layer_analysis;
        layerHtml += `
        <table class="data-table">
            <tr><th>Component</th><th>Percentage</th></tr>
            <tr><td>Backbone</td><td>${la.components.backbone.percentage.toFixed(1)}%</td></tr>
            <tr><td>Decoder</td><td>${la.components.decoder.percentage.toFixed(1)}%</td></tr>
            <tr><td><br></td><td></td></tr>
            <tr><td><strong>Total Params</strong></td><td><strong>${(la.total_params / 1e6).toFixed(2)} M</strong></td></tr>
        </table>
        `;
    } else {
        layerHtml += '<div style="padding:10px; color:var(--text-sub)">No layer profiling data available</div>';
    }
    document.getElementById('tab-layers').innerHTML = layerHtml;
}

window.switchTab = function (tabName) {
    // Hide all content
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    // Show target content
    document.getElementById(`tab-${tabName}`).classList.remove('hidden');

    // Highlight button
    document.getElementById(`btn-${tabName}`).classList.add('active');
};

// Upload Handling
dom.uploadZone.addEventListener('click', () => dom.fileInput.click());
dom.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dom.uploadZone.style.borderColor = 'var(--primary)';
});
dom.uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dom.uploadZone.style.borderColor = '';
});
dom.uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.uploadZone.style.borderColor = '';
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

dom.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

async function handleFile(file) {
    if (!currentModelId) return alert("Please select a model first.");

    // Show Preview
    await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            dom.originalPreview.src = e.target.result;
            dom.resultPreview.src = ""; // Clear previous
            dom.previewContainer.classList.remove('hidden');
            dom.previewContainer.classList.add('visible');
            resolve();
        };
        reader.readAsDataURL(file);
    });

    // Upload & Predict
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', currentModelId);

    dom.loader.classList.remove('hidden');

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Inference failed (${res.status}): ${errText}`);
        }

        const data = await res.json();

        if (!data.annotated_image) {
            throw new Error("Backend returned no image data");
        }

        lastResult = data;

        // Display result
        dom.resultPreview.src = `data:image/jpeg;base64,${data.annotated_image}`;
        dom.resultPreview.onerror = null;

        dom.downloadControls.classList.remove('hidden');

    } catch (e) {
        console.error(e);
        alert("Error during inference: " + e.message);
    } finally {
        dom.loader.classList.add('hidden');
    }
}

function downloadResult(type) {
    if (!lastResult) return;

    if (type === 'image') {
        const a = document.createElement('a');
        a.href = `data:image/jpeg;base64,${lastResult.annotated_image}`;
        a.download = `prediction_${Date.now()}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    } else if (type === 'json') {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastResult.detections, null, 2));
        const a = document.createElement('a');
        a.href = dataStr;
        a.download = `detections_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}

function downloadMetrics(type) {
    if (!currentModelId) return alert("Please select a model first.");
    const model = modelsData.find(m => m.id === currentModelId);
    if (!model) return;

    const timestamp = new Date().toISOString().split('T')[0];
    const filename = `${model.id}_metrics_${timestamp}`;

    if (type === 'json') {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(model.full_metrics, null, 2));
        const a = document.createElement('a');
        a.href = dataStr;
        a.download = `${filename}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    } else if (type === 'csv') {
        let csvContent = "data:text/csv;charset=utf-8,";

        // 1. Summary Header
        csvContent += "Metric,Value\n";

        // Add Summary Metrics
        Object.entries(model.summary_metrics).forEach(([key, value]) => {
            csvContent += `${key},${value}\n`;
        });

        // Add FPS/Latency if available
        const fps = model.full_metrics.fps?.fps_benchmark?.['640x640'];
        if (fps) {
            csvContent += `FPS Mean,${fps.fps_mean}\n`;
            csvContent += `Latency Mean (ms),${fps.latency_mean_ms}\n`;
        }

        // 2. Class Metrics Header
        csvContent += "\nClass Metrics\n";
        csvContent += "Class Name,AP 50,AP 50-95,AP Small\n";

        // Add Class Rows
        if (model.full_metrics.detailed?.per_class_metrics) {
            Object.values(model.full_metrics.detailed.per_class_metrics).forEach(m => {
                csvContent += `${m.name},${m.ap_50},${m.ap_50_95},${m.ap_small}\n`;
            });
        }

        // 3. Layer Analysis
        csvContent += "\nLayer Analysis\n";
        csvContent += "Component,Params\n";
        if (model.full_metrics.profiling?.layer_analysis) {
            const la = model.full_metrics.profiling.layer_analysis;
            csvContent += `Total Params,${la.total_params}\n`;
            csvContent += `Backbone Params,${la.components.backbone.total_params}\n`;
            csvContent += `Decoder Params,${la.components.decoder.total_params}\n`;
        }

        const encodedUri = encodeURI(csvContent);
        const a = document.createElement('a');
        a.href = encodedUri;
        a.download = `${filename}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}

// Start
init();
