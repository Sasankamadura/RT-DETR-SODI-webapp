from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import io
import base64
import time
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from .model_utils import ModelHandler

app = FastAPI(title="RT-DETR Research Prototype")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sample images
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Sample Visdrone Images")
if os.path.exists(SAMPLES_DIR):
    app.mount("/samples-data", StaticFiles(directory=SAMPLES_DIR), name="samples")
else:
    print(f"Warning: Samples directory not found at {SAMPLES_DIR}")

# Load configuration
# Load configuration path
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "models_config.json")

def get_config():
    """Load latest configuration from disk."""
    if not os.path.exists(CONFIG_PATH):
        return []
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

# Global model cache (lazy loading)
loaded_models = {}

@app.get("/models")
async def get_models():
    """Return list of available models with metadata and metrics."""
    return get_config()

@app.post("/predict")
async def predict(
    model_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None)
):
    """Run inference on uploaded image or server-side sample."""
    
    # Reload config to get latest paths/metadata
    models_config = get_config()

    # optimize: check if model config exists
    target_config = next((m for m in models_config if m['id'] == model_id), None)
    if not target_config:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # optimize: lazy load model
    if model_id not in loaded_models:
        print(f"Loading model: {target_config['name']}")
        try:
            # Construct absolute path relative to current dir
            # config path is relative to repo root, let's fix it
            # assuming running from repo root
            model_path = target_config['path']
            loaded_models[model_id] = ModelHandler(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
            
    # Read image
    image_bytes = None
    if sample_filename:
        # Securely join path (prevent traversal)
        safe_filename = os.path.basename(sample_filename)
        sample_path = os.path.join(SAMPLES_DIR, safe_filename)
        if not os.path.exists(sample_path):
             raise HTTPException(status_code=404, detail=f"Sample file not found: {sample_path}")
        
        print(f"Loading sample from disk: {sample_path}")
        with open(sample_path, "rb") as f:
            image_bytes = f.read()
    elif file:
        image_bytes = await file.read()
    else:
        raise HTTPException(status_code=400, detail="Either 'file' or 'sample_filename' must be provided.")
    
    
    try:
        start_time = time.time()
        handler = loaded_models[model_id]
        annotated_image, detections = handler.predict(image_bytes)
        inference_time = time.time() - start_time
        
        # Determine original filename for display if possible (not possible from upload stream directly efficiently without client header, using generic)
        
        # Convert annotated image to base64
        img_byte_arr = io.BytesIO()
        annotated_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        encoded_image = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        
        return JSONResponse({
            "annotated_image": encoded_image,
            "detections": detections,
            "inference_time": inference_time
        })
        
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/samples")
async def get_samples():
    """List available sample images."""
    if not os.path.exists(SAMPLES_DIR):
        return []
    
    files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return [{"filename": f, "url": f"/samples-data/{f}"} for f in files]

# Serve Frontend (Must be last to avoid overriding API routes)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory not found at {FRONTEND_DIR}")
