# RT-DETR Research Prototype: Small Object Detection Enhancement

![Project Status](https://img.shields.io/badge/Status-Research_Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![ONNX](https://img.shields.io/badge/Inference-ONNX_Runtime-blueviolet)

A full-stack web application demonstrating enhancements to the RT-DETR architecture, specifically targeting **Small Object Detection (SOD)** in drone imagery (VisDrone2019 dataset).

This repository contains the deployment code for serving 4 variants of the RT-DETR ResNet-18 model, complete with a glassmorphic user interface for real-time inference and metric analysis.

## \u2728 Key Features

*   **Multi-Model Interface**: Seamlessly switch between the Baseline RT-DETR and 3 improved research variants.
*   **Real-Time Inference**: Drag-and-drop image upload with instant bounding box generation using ONNX Runtime.
*   **Detailed Metrics Dashboard**: Interactive tabs displaying:
    *   **Class Metrics**: Per-class AP scores (e.g., Pedestrian, Bicycle).
    *   **Layer Analysis**: Parameter distribution between Backbone and Decoder.
    *   **Inference Stats**: Latency and FPS benchmarks.
*   **Premium UI**: Fully responsive, dark-themed glassmorphism design.

## \u1f680 The Models

This application compares four model variants trained on the **VisDrone2019** dataset:

| Model ID | Variant Name | Key Innovation | mAP (50) | FPS (GPU) |
| :--- | :--- | :--- | :--- | :--- |
| `base_rtdetr` | **Baseline** | Standard RT-DETR-R18 | *Baseline* | *N/A* |
| `p2_p3_fusion` | **P2-P3 Fusion** | High-resolution feature fusion | **41.18%** | ~60 |
| `query_imp` | **Query IMP** | Scale-aware query initialization | **36.74%** | ~58 |
| `aware_loss` | **Aware Loss** | Difficulty-aware loss reweighting| **18.85%** | ~60 |

## \u1f6e0\ufe0f Tech Stack

*   **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), JavaScript (ES6).
*   **Backend**: Python FastAPI.
*   **ML Inference**: ONNX Runtime (GPU/CPU).
*   **Containerization**: Docker.

## \u1f4bb Quick Start

### Option A: Docker (Recommended)
You can build and run the entire application with a single command:

```bash
docker build -t rtdetr-app .
docker run -p 8000:8000 rtdetr-app
```
Access the app at `http://localhost:8000`.

### Option B: Local Development
1.  **Clone the repository** (Ensure you use Git LFS for models):
    ```bash
    git clone https://github.com/YourUsername/rt-detr-webapp.git
    cd rt-detr-webapp
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    ```

3.  **Run Server**:
    ```bash
    python run.py
    ```

4.  **Open in Browser**:
    Navigate to `http://localhost:8000/frontend/index.html` (or `http://localhost:8000` if using the unified serving method).

## \u1f4c2 Project Structure

```
\u251c\u2500\u2500 backend/
\u2502   \u251c\u2500\u2500 models/          # .onnx model files (tracked by LFS)
\u2502   \u251c\u2500\u2500 main.py          # FastAPI application
\u2502   \u251c\u2500\u2500 model_utils.py   # Preprocessing & Inference logic
\u2502   \u2514\u2500\u2500 models_config.json # Auto-generated metrics DB
\u251c\u2500\u2500 frontend/
\u2502   \u251c\u2500\u2500 index.html       # Main UI
\u2502   \u251c\u2500\u2500 style.css        # Premium styling
\u2502   \u2514\u2500\u2500 script.js        # UI Logic & API calls
\u251c\u2500\u2500 Dockerfile           # Deployment configuration
\u251c\u2500\u2500 run.py               # Local execution script
\u2514\u2500\u2500 requirements.txt     # Python dependencies
```

## \u26a0\ufe0f Note on Large Files
This repository uses **Git LFS** to store the `.onnx` model files (~300MB each). Ensure you have Git LFS installed before cloning.

---
*Created for Research Prototype Demonstration.*
