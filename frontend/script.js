/**
 * RT-DETR Model Explorer - OOP Implementation
 */

// --- Constants & Config ---
const CONFIG = {
    API_URL: (window.location.protocol === 'file:' || window.location.port === '5500')
        ? "http://localhost:8000"
        : ""
};

/**
 * Service Layer: Handles API communication
 */
class ModelService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async fetchModels() {
        // Appending timestamp to strictly bypass aggressive browser caching
        const res = await fetch(`${this.baseUrl}/models?t=${new Date().getTime()}`, {
            cache: 'no-store'
        });
        if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
        return await res.json();
    }

    async fetchSamples() {
        const res = await fetch(`${this.baseUrl}/samples`);
        if (!res.ok) throw new Error(`Failed to fetch samples: ${res.statusText}`);
        return await res.json();
    }

    async predict(formData) {
        const res = await fetch(`${this.baseUrl}/predict`, {
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
        return data;
    }
}

/**
 * UI Layer: Handles DOM manipulation and Rendering
 */
class UIManager {
    constructor() {
        this.dom = {
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
            samplesContainer: document.getElementById('samples-grid'),
            downloadControls: document.getElementById('download-controls'),

            // Tabs
            tabs: {
                overview: document.getElementById('tab-overview'),
                classMetrics: document.getElementById('tab-class-metrics'),
                layers: document.getElementById('tab-layers')
            },
            tabButtons: document.querySelectorAll('.tab-btn'),
            tabContents: document.querySelectorAll('.tab-content')
        };
    }

    // --- Rendering Methods ---

    renderModelList(models, onSelect) {
        this.dom.modelList.innerHTML = models.map(model => {
            const isExperimental = model.version === 'Experimental';
            return `
                <div class="model-item ${isExperimental ? 'experimental' : ''}" data-id="${model.id}" id="model-${model.id}">
                    <span class="model-name">${model.name}</span>
                </div>
            `;
        }).join('');

        // Add event listeners locally to avoid global pollution
        this.dom.modelList.querySelectorAll('.model-item').forEach(item => {
            item.addEventListener('click', () => onSelect(item.dataset.id));
        });
    }

    setActiveModel(id) {
        this.dom.modelList.querySelectorAll('.model-item').forEach(el => el.classList.remove('active'));
        const activeItem = document.getElementById(`model-${id}`);
        if (activeItem) activeItem.classList.add('active');
    }

    updateModelDetails(model) {
        this.dom.currentModelName.textContent = model.name;
        this.dom.description.innerHTML = `<p>${model.description}</p>`;

        // Render Summary Metrics
        this.dom.metricsGrid.innerHTML = Object.entries(model.summary_metrics).map(([key, value]) => `
            <div class="metric-card">
                <span class="label">${key.replace('_', ' ')}</span>
                <span class="value" style="color: ${value !== 'N/A' ? '#10b981' : '#9ca3af'}">${value}</span>
            </div>
        `).join('');

        this.renderDetailedMetrics(model);
    }

    renderDetailedMetrics(model) {
        const full = model.full_metrics;

        // 1. Overview Tab
        let overviewHtml = '<table class="data-table"><tr><th>Metric</th><th>Value</th></tr>';

        if (full.profiling?.layer_analysis) {
            const totalParams = full.profiling.layer_analysis.total_params;
            overviewHtml += `
                <tr><td>Total Params</td><td>${(totalParams / 1e6).toFixed(2)} M</td></tr>
                <tr><td>Trainable</td><td>${(totalParams / 1e6).toFixed(2)} M</td></tr>
            `;
        } else {
            overviewHtml += `<tr><td>Params</td><td>N/A</td></tr>`;
        }

        const bench = full.fps?.fps_benchmark?.['640x640'];
        if (bench) {
            overviewHtml += `
                <tr><td>Latency Mean</td><td>${bench.latency_mean_ms.toFixed(2)} ms</td></tr>
                <tr><td>Latency Min</td><td>${bench.latency_min_ms.toFixed(2)} ms</td></tr>
                <tr><td>FPS Mean</td><td>${bench.fps_mean.toFixed(2)}</td></tr>
            `;
        }
        overviewHtml += '</table>';
        this.dom.tabs.overview.innerHTML = overviewHtml;

        // 2. Class Metrics Tab
        let classHtml = '<table class="data-table"><tr><th>Class</th><th>AP 50</th><th>AP 50-95</th></tr>';
        if (full.detailed?.per_class_metrics) {
            Object.entries(full.detailed.per_class_metrics).forEach(([className, m]) => {
                const ap50 = (m.AP_50 * 100).toFixed(1);
                const ap50_95 = (m.AP_50_95 * 100).toFixed(1);
                classHtml += `<tr>
                    <td>${className}</td>
                    <td>${ap50}%</td>
                    <td>${ap50_95}%</td>
                </tr>`;
            });
        } else {
            classHtml += '<tr><td colspan="3">No detailed class metrics available</td></tr>';
        }
        classHtml += '</table>';
        this.dom.tabs.classMetrics.innerHTML = classHtml;

        // 3. Layer Analysis Tab
        let layerHtml = '';
        if (full.profiling?.layer_analysis) {
            const la = full.profiling.layer_analysis;
            layerHtml += `
            <table class="data-table">
                <tr><th>Component</th><th>Percentage</th></tr>
                <tr><td>Backbone</td><td>${la.components.backbone ? la.components.backbone.percentage.toFixed(1) : 0}%</td></tr>
                ${la.components.encoder ? `<tr><td>Encoder</td><td>${la.components.encoder.percentage.toFixed(1)}%</td></tr>` : ''}
                <tr><td>Decoder</td><td>${la.components.decoder ? la.components.decoder.percentage.toFixed(1) : 0}%</td></tr>
                <tr><td><br></td><td></td></tr>
                <tr><td><strong>Total Params</strong></td><td><strong>${(la.total_params / 1e6).toFixed(2)} M</strong></td></tr>
            </table>
            `;
        } else {
            layerHtml += '<div style="padding:10px; color:var(--text-sub)">No layer profiling data available</div>';
        }
        this.dom.tabs.layers.innerHTML = layerHtml;
    }

    renderSampleList(samples, onSelect) {
        this.dom.samplesContainer.innerHTML = samples.map(s => `
            <div class="sample-item" data-url="${s.url}">
                <img src="${CONFIG.API_URL}${s.url}" alt="${s.filename}" loading="lazy">
            </div>
        `).join('');

        this.dom.samplesContainer.querySelectorAll('.sample-item').forEach(item => {
            item.addEventListener('click', () => onSelect(item.dataset.url));
        });
    }

    switchTab(tabName) {
        this.dom.tabContents.forEach(el => el.classList.add('hidden'));
        this.dom.tabButtons.forEach(el => el.classList.remove('active'));

        const targetTab = document.getElementById(`tab-${tabName}`);
        const targetBtn = document.getElementById(`btn-${tabName}`);

        if (targetTab) targetTab.classList.remove('hidden');
        if (targetBtn) targetBtn.classList.add('active');
    }

    // --- Preview & Results ---

    resetPreview() {
        this.dom.fileInput.value = "";
        this.dom.previewContainer.classList.remove('visible');
        this.dom.previewContainer.classList.add('hidden');
        this.dom.originalPreview.src = "";
        this.dom.resultPreview.src = "";
        this.dom.uploadZone.style.borderColor = "";
        this.dom.downloadControls.classList.add('hidden');
    }

    showOriginal(src) {
        this.dom.originalPreview.src = src;
        this.dom.resultPreview.src = "";
        this.dom.previewContainer.classList.remove('hidden');
        this.dom.previewContainer.classList.add('visible');
        // Clear previous result handlers if any
        this.dom.resultPreview.onerror = null;
    }

    showResult(base64Image) {
        this.dom.resultPreview.src = `data:image/jpeg;base64,${base64Image}`;
        this.dom.downloadControls.classList.remove('hidden');
    }

    showLoader(show) {
        if (show) this.dom.loader.classList.remove('hidden');
        else this.dom.loader.classList.add('hidden');
    }

    showError(container, message) {
        if (container) {
            container.innerHTML = `<p style="color:red; text-align:center">${message}</p>`;
        } else {
            alert(message);
        }
    }
}

/**
 * Main Application Controller
 */
class RTDETRApp {
    constructor() {
        this.service = new ModelService(CONFIG.API_URL);
        this.ui = new UIManager();

        this.state = {
            currentModelId: null,
            modelsData: [],
            lastResult: null
        };
    }

    async init() {
        try {
            await this.loadModels();
            await this.loadSamples();
            this.bindEvents();
        } catch (e) {
            console.error(e);
            this.ui.showError(this.ui.dom.modelList, "Error initializing application");
        }
    }

    bindEvents() {
        // Tab Switching
        // Expose switchTab globally if needed by HTML onclicks, or bind here
        // The original HTML uses `onclick="switchTab(...)"`, so we keep the global function
        // or we can attach listeners if we have references.
        // For cleaner OOP, we should attach listeners, but let's support the existing HTML structure.
        window.switchTab = (name) => this.ui.switchTab(name);

        // Upload Zone
        const dz = this.ui.dom.uploadZone;
        dz.addEventListener('click', () => this.ui.dom.fileInput.click());
        dz.addEventListener('dragover', (e) => {
            e.preventDefault();
            dz.style.borderColor = 'var(--primary)';
        });
        dz.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dz.style.borderColor = '';
        });
        dz.addEventListener('drop', (e) => {
            e.preventDefault();
            dz.style.borderColor = '';
            const file = e.dataTransfer.files[0];
            if (file) this.handleFile(file);
        });

        this.ui.dom.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) this.handleFile(file);
        });

        // Downloads - Modify HTML to remove onclicks or attach listeners if IDs exist
        // The HTML likely has onclick="downloadResult(...)". We should override/attach.
        // Since we are rewriting script.js, existing HTML onclicks will fail unless functions are global.
        // Strategy: Attach listeners to known IDs or expose global wrappers.
        // Let's expose global wrappers for compatibility with existing HTML.
        window.downloadResult = (type) => this.downloadResult(type);
        window.downloadMetrics = (type) => this.downloadMetrics(type);
    }

    // --- Logic Methods ---

    async loadModels() {
        this.state.modelsData = await this.service.fetchModels();
        this.ui.renderModelList(this.state.modelsData, (id) => this.selectModel(id));

        if (this.state.modelsData.length > 0) {
            this.selectModel(this.state.modelsData[0].id);
        }
    }

    selectModel(id) {
        this.state.currentModelId = id;
        this.ui.setActiveModel(id);

        const model = this.state.modelsData.find(m => m.id === id);
        if (model) {
            this.ui.updateModelDetails(model);
        }

        this.ui.resetPreview();
        this.state.lastResult = null;
    }

    async loadSamples() {
        try {
            const samples = await this.service.fetchSamples();
            this.ui.renderSampleList(samples, (url) => this.selectSample(url));
        } catch (e) {
            this.ui.showError(this.ui.dom.samplesContainer, "Failed to load samples");
        }
    }

    async selectSample(url) {
        if (!this.state.currentModelId) return alert("Please select a model first.");

        const filename = decodeURIComponent(url.split('/').pop());
        this.ui.showOriginal(`${CONFIG.API_URL}${url}`);
        this.ui.showLoader(true);

        const formData = new FormData();
        formData.append('model_id', this.state.currentModelId);
        formData.append('sample_filename', filename);

        try {
            const data = await this.service.predict(formData);
            this.state.lastResult = data;
            this.ui.showResult(data.annotated_image);
        } catch (e) {
            console.error(e);
            alert("Error: " + e.message);
        } finally {
            this.ui.showLoader(false);
        }
    }

    async handleFile(file) {
        if (!this.state.currentModelId) return alert("Please select a model first.");

        // Preview
        const reader = new FileReader();
        reader.onload = (e) => this.ui.showOriginal(e.target.result);
        reader.readAsDataURL(file);

        this.ui.showLoader(true);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_id', this.state.currentModelId);

        try {
            const data = await this.service.predict(formData);
            this.state.lastResult = data;
            this.ui.showResult(data.annotated_image);
        } catch (e) {
            console.error(e);
            alert("Error: " + e.message);
        } finally {
            this.ui.showLoader(false);
        }
    }

    // --- Utilities ---

    downloadResult(type) {
        if (!this.state.lastResult) return;

        if (type === 'image') {
            this._downloadFile(`prediction_${Date.now()}.jpg`, `data:image/jpeg;base64,${this.state.lastResult.annotated_image}`);
        } else if (type === 'json') {
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(this.state.lastResult.detections, null, 2));
            this._downloadFile(`detections_${Date.now()}.json`, dataStr);
        }
    }

    downloadMetrics(type) {
        if (!this.state.currentModelId) return alert("Please select a model first.");
        const model = this.state.modelsData.find(m => m.id === this.state.currentModelId);
        if (!model) return;

        const timestamp = new Date().toISOString().split('T')[0];
        const filename = `${model.id}_metrics_${timestamp}`;

        if (type === 'json') {
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(model.full_metrics, null, 2));
            this._downloadFile(`${filename}.json`, dataStr);
        } else if (type === 'csv') {
            const csvContent = this._generateCSV(model);
            const encodedUri = encodeURI("data:text/csv;charset=utf-8," + csvContent);
            this._downloadFile(`${filename}.csv`, encodedUri);
        }
    }

    _downloadFile(filename, linkSource) {
        const a = document.createElement('a');
        a.href = linkSource;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    _generateCSV(model) {
        let csv = "Metric,Value\n";

        // Summary
        Object.entries(model.summary_metrics).forEach(([key, value]) => {
            csv += `${key},${value}\n`;
        });

        // FPS
        const fps = model.full_metrics.fps?.fps_benchmark?.['640x640'];
        if (fps) {
            csv += `FPS Mean,${fps.fps_mean}\n`;
            csv += `Latency Mean (ms),${fps.latency_mean_ms}\n`;
        }

        // Class Metrics
        csv += "\nClass Metrics\nClass Name,AP 50,AP 50-95\n";
        if (model.full_metrics.detailed?.per_class_metrics) {
            Object.entries(model.full_metrics.detailed.per_class_metrics).forEach(([className, m]) => {
                csv += `${className},${m.AP_50},${m.AP_50_95}\n`;
            });
        }

        // Layers
        csv += "\nLayer Analysis\nComponent,Params\n";
        if (model.full_metrics.profiling?.layer_analysis) {
            const la = model.full_metrics.profiling.layer_analysis;
            csv += `Total Params,${la.total_params}\n`;
            if (la.components.backbone) csv += `Backbone Params,${la.components.backbone.total_params}\n`;
            if (la.components.encoder) csv += `Encoder Params,${la.components.encoder.total_params}\n`;
            if (la.components.decoder) csv += `Decoder Params,${la.components.decoder.total_params}\n`;
        }
        return csv;
    }
}

// --- Initialize App ---
const app = new RTDETRApp();
// Expose app logic mainly for event binders if needed, 
// though we handle most internally or via window wrappers.
document.addEventListener('DOMContentLoaded', () => {
    app.init();
});
