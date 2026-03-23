# RT-DETR Research Prototype: Small Object Detection (SOD)

![Project Status](https://img.shields.io/badge/Status-Research_Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![ONNX](https://img.shields.io/badge/Inference-ONNX_Runtime-blueviolet)

This repository demonstrates advanced architectural enhancements to the **RT-DETR** (Real-Time DEtection TRansformer) model, specifically optimized for **Small Object Detection (SOD)** in drone-captured aerial imagery using the **VisDrone2019** dataset.

The prototype features a high-performance FastAPI/ONNX backend and a premium glassmorphic web interface for real-time inference and comparative analysis of multiple research variants.

## ✨ Key Features

*   **Multi-Model Dashboard**: Compare 10+ research variants including Baseline, P2-enhanced, GnConv-integrated, and EfficientNet backbones.
*   **Real-Time Inference**: Drag-and-drop imagery for instant object detection with sub-millisecond latency visualization.
*   **Deep Metric Analysis**: Interactive tabs for Class-wise Precision (AP), Parameter Distribution, and GFLOPs profiling.
*   **Research Visibility**: Comprehensive visualization of backbone vs. encoder/decoder complexity.
*   **Premium Web Experience**: Fully responsive, dark-themed glassmorphism UI with micro-animations.

## 🚀 Comparison of Final Models

The following performance metrics were verified on the **VisDrone2019** dataset after 101 epochs of training:

| Model Variant | Backbone | Innovation | mAP (50) | mAP (Small) | FPS (GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | ResNet-18 | **Reference Model** | 42.33% | 16.29% | 164.36 |
| **EfficientNet + GnConv + P2** | EfficientNet-B2 | Backbone swap + GnConv Encoder + P2 | **48.29%** | **20.13%** | 75.05 |
| **GnConv + P2 Layer** | ResNet-18 | GnConv-based High-Res Fusion | 45.76% | 18.96% | 91.61 |
| **P2 Layer (Fusion)** | ResNet-18 | Standard RepVGG Fusion | 46.82% | 19.75% | 113.96 |
| **Slim P2 Layer** | ResNet-18 | Efficient RepVGG Fusion | 45.18% | 18.72% | 122.03 |
| **GnConv + SLIM P2** | ResNet-18 | Balanced Efficiency/Resolution | 44.41% | 18.05% | 106.39 |
| **Final Improved Gnconv** | ResNet-18 | Parallel Pooling & Gating | 44.16% | 18.17% | 86.94 |
| **Final Query Improvement** | ResNet-18 | Scale-aware Query Selection | 43.83% | 16.74% | 160.21 |

## 🛠️ Tech Stack

*   **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), JavaScript (ES6).
*   **Backend**: Python FastAPI (Async Engine).
*   **ML Inference**: ONNX Runtime (GPU-accelerated via CUDA).
*   **Containerization**: Docker (Production-ready).

## 💻 Running the App

### Option A: Local Development
1.  **Clone the repository**:
    ```bash
    git clone [repository-url]
    cd [repository-name]
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    ```
3.  **Run Server**:
    ```bash
    python run.py
    ```
4.  **Access App**:
    Navigate to `http://localhost:8000`.

### Option B: Docker
```bash
docker build -t rtdetr-research .
docker run -p 8000:8000 rtdetr-research
```

## 📂 Project Structure

*   `backend/`: FastAPI application, model handlers, and `models_config.json`.
*   `frontend/`: Source code for the glassmorphic web UI.
*   `Normal RT-DETR/`: Core model architecture and local inference test scripts.
*   `Final Models/`: Comprehensive benchmarks and validation reports.
*   `kaggle_scripts/`: Training scripts optimized for Kaggle/Colab environments.
*   `Sample Visdrone Images/`: Curated dataset samples for testing.
*   `run.py`: Entry point for starting the unified server.

---
*Developed for RT-DETR Research and Small Object Detection Profiling.*
