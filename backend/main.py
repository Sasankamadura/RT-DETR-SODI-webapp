from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import io
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
    file: UploadFile = File(...)
):
    """Run inference on uploaded image."""
    
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
    image_bytes = await file.read()
    
    try:
        handler = loaded_models[model_id]
        annotated_image = handler.predict(image_bytes)
        
        # Return image
        img_byte_arr = io.BytesIO()
        annotated_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
        
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "RT-DETR Backend Running"}
