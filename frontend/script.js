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
    description: document.getElementById('model-description')
};

let currentModelId = null;
let modelsData = [];

// Init
async function init() {
    try {
        const res = await fetch(`${API_URL}/models`);
        modelsData = await res.json();
        renderModelList(modelsData);
        if (modelsData.length > 0) {
            selectModel(modelsData[0].id);
        }
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
}

function renderDetailedMetrics(model) {
    const full = model.full_metrics;

    // 1. Overview Tab
    let overviewHtml = '<table class="data-table"><tr><th>Metric</th><th>Value</th></tr>';
    if (full.evaluation) {
        overviewHtml += `
            <tr><td>Model Size</td><td>${full.evaluation.model_size_mb.toFixed(2)} MB</td></tr>
            <tr><td>Params</td><td>${full.evaluation.model_keys}</td></tr>
        `;
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
    let layerHtml = '<div style="padding:10px">';
    if (full.profiling && full.profiling.layer_analysis) {
        const la = full.profiling.layer_analysis;
        layerHtml += `<p>Total Params: ${la.total_params.toLocaleString()}</p>`;
        layerHtml += `<p>Backbone: ${la.components.backbone.percentage.toFixed(1)}%</p>`;
        layerHtml += `<p>Decoder: ${la.components.decoder.percentage.toFixed(1)}%</p>`;
    } else {
        layerHtml += '<p>No layer profiling data available</p>';
    }
    layerHtml += '</div>';
    document.getElementById('tab-layers').innerHTML = layerHtml;
}

window.switchTab = function (tabName) {
    // Hide all
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    // Show target
    document.getElementById(`tab-${tabName}`).classList.remove('hidden');
    // Highlight button (simplistic matching)
    const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.textContent.toLowerCase().includes(tabName.split('-')[0]));
    if (btn) btn.classList.add('active');
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
    const reader = new FileReader();
    reader.onload = (e) => {
        dom.originalPreview.src = e.target.result;
        dom.resultPreview.src = ""; // Clear previous
        dom.previewContainer.classList.remove('hidden');
        dom.previewContainer.classList.add('visible');
    };
    reader.readAsDataURL(file);

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

        if (!res.ok) throw new Error("Inference failed");

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        dom.resultPreview.src = url;
    } catch (e) {
        console.error(e);
        alert("Error during inference: " + e.message);
    } finally {
        dom.loader.classList.add('hidden');
    }
}

// Start
init();
