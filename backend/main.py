from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import io
import base64
import time
import traceback
from .model_utils import ModelHandler

class RTDETRBackendApp:
    def __init__(self):
        self.app = FastAPI(title="RT-DETR Research Prototype")
        self.loaded_models = {}
        self.config_path = os.path.join(os.path.dirname(__file__), "models_config.json")
        self.samples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Sample Visdrone Images")
        self.frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
        
        self._configure_middleware()
        self._mount_static_files()
        self._setup_routes()

    def _configure_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _mount_static_files(self):
        # Mount sample images
        if os.path.exists(self.samples_dir):
            self.app.mount("/samples-data", StaticFiles(directory=self.samples_dir), name="samples")
        else:
            print(f"Warning: Samples directory not found at {self.samples_dir}")

        # Mount Frontend (Must be last to avoid overriding API routes)
        # We will mount this AFTER routes or ensure explicit routes take precedence.
        # But FastAPI mounts match by longest prefix so explicit paths usually win.
        # However, mounting "/" usually catches everything not matched.
        # Better to do this in _setup_routes or at the end of __init__.
        pass # Will do at end of setup

    def _setup_routes(self):
        # Define routes with class methods as handlers
        self.app.get("/models")(self.get_models)
        self.app.post("/predict")(self.predict)
        self.app.get("/samples")(self.get_samples)

        # Mount frontend last
        if os.path.exists(self.frontend_dir):
            self.app.mount("/", StaticFiles(directory=self.frontend_dir, html=True), name="frontend")
        else:
            print(f"Warning: Frontend directory not found at {self.frontend_dir}")

    def get_config(self):
        """Load latest configuration from disk."""
        if not os.path.exists(self.config_path):
            return []
        with open(self.config_path, "r") as f:
            return json.load(f)

    async def get_models(self):
        """Return list of available models with metadata and metrics."""
        return self.get_config()

    async def get_samples(self):
        """List available sample images."""
        if not os.path.exists(self.samples_dir):
            return []
        
        files = [f for f in os.listdir(self.samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return [{"filename": f, "url": f"/samples-data/{f}"} for f in files]

    async def predict(self, 
                      model_id: str = Form(...), 
                      file: Optional[UploadFile] = File(None), 
                      sample_filename: Optional[str] = Form(None)):
        """Run inference on uploaded image or server-side sample."""
        try:
            start_time = time.time()
            
            # Get Handler
            handler = self._get_model_handler(model_id)
            
            # Read Image
            image_bytes = await self._read_image(file, sample_filename)
            
            # Run Inference
            annotated_image, detections = handler.predict(image_bytes)
            inference_time = time.time() - start_time
            
            # Encode Response
            encoded_image = self._encode_image(annotated_image)
            
            return JSONResponse({
                "annotated_image": encoded_image,
                "detections": detections,
                "inference_time": inference_time
            })
            
        except HTTPException as he:
            raise he
        except Exception as e:
            print(f"Inference error: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    def _get_model_handler(self, model_id: str):
        # Lazy load model
        if model_id not in self.loaded_models:
            models_config = self.get_config()
            target_config = next((m for m in models_config if m['id'] == model_id), None)
            
            if not target_config:
                raise HTTPException(status_code=404, detail="Model not found")
            
            print(f"Loading model: {target_config['name']}")
            try:
                model_path = target_config['path']
                version = target_config.get('version', 'Final')
                indexing_type = "0-indexed" if version == "Final" else "1-indexed"
                
                self.loaded_models[model_id] = ModelHandler(model_path, indexing_type=indexing_type)
            except Exception as e:
                print(f"Error loading model: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        return self.loaded_models[model_id]

    async def _read_image(self, file: Optional[UploadFile], sample_filename: Optional[str]) -> bytes:
        if sample_filename:
            safe_filename = os.path.basename(sample_filename)
            sample_path = os.path.join(self.samples_dir, safe_filename)
            if not os.path.exists(sample_path):
                 raise HTTPException(status_code=404, detail=f"Sample file not found: {sample_path}")
            
            print(f"Loading sample from disk: {sample_path}")
            with open(sample_path, "rb") as f:
                return f.read()
        elif file:
            return await file.read()
        else:
            raise HTTPException(status_code=400, detail="Either 'file' or 'sample_filename' must be provided.")

    def _encode_image(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return base64.b64encode(img_byte_arr.read()).decode('utf-8')

# Initialize App
backend_app = RTDETRBackendApp()
app = backend_app.app
