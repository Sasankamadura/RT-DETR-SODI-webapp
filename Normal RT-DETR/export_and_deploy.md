# Local ONNX Deployment Guide - RT-DETR R18

## Step 1: Export Model to ONNX (Run on Kaggle)

Since your checkpoint is on Kaggle, export it there first:

```bash
# On Kaggle notebook
!python tools/export_onnx.py \
    -c /kaggle/working/RT-DETR_SOD_IMP/rtdetr_pytorch/configs/rtdetr/rtdetr_visdrone_r18vd.yml \
    -r /kaggle/input/rt-detr-r18-baseline-on-visdrone2019-c/checkpoint0093.pth \
    --check

# This creates: rtdetr_r18.onnx (~90MB)
```

**Download** the generated `rtdetr_r18.onnx` file from Kaggle to your local PC.

---

## Step 2: Setup Python Environment (Your PC - Python 3.12)

### Install Dependencies

```powershell
# Open PowerShell and run:
pip install onnxruntime-gpu opencv-python pillow numpy
```

**Or for CPU-only** (if no NVIDIA GPU):
```powershell
pip install onnxruntime opencv-python pillow numpy
```

### Verify Installation

```python
import onnxruntime as ort
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")
```

Expected output:
```
ONNX Runtime version: 1.17.x
Available providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## Step 3: Create Inference Script (Your PC)

Save this as `local_inference.py` on your PC:

```python
# local_inference.py
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

# Configuration
MODEL_PATH = "rtdetr_r18.onnx"  # Path to your downloaded ONNX model
INPUT_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.4

# VisDrone class names
CLASS_NAMES = {
    0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van',
    5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'
}

class RTDETRInference:
    def __init__(self, model_path, use_gpu=True):
        """Initialize ONNX Runtime session."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        print(f"Model loaded successfully!")
        print(f"Using: {self.session.get_providers()[0]}")
    
    def preprocess(self, image_path):
        """Load and preprocess image."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Resize to model input size
        image_resized = image.resize(INPUT_SIZE)
        
        # Convert to numpy array and normalize
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        
        # Transpose to CHW format
        image_np = image_np.transpose(2, 0, 1)
        
        # Add batch dimension
        image_batch = np.expand_dims(image_np, 0)
        
        return image, image_batch, original_size
    
    def infer(self, image_batch):
        """Run inference."""
        outputs = self.session.run(None, {self.input_name: image_batch})
        return outputs
    
    def postprocess(self, outputs, original_size, threshold=CONFIDENCE_THRESHOLD):
        """Process model outputs."""
        labels = outputs[0][0]  # Shape: (num_detections,)
        boxes = outputs[1][0]   # Shape: (num_detections, 4) - [x1, y1, x2, y2]
        scores = outputs[2][0]  # Shape: (num_detections,)
        
        # Filter by confidence
        mask = scores > threshold
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]
        
        # Scale boxes to original image size
        # boxes are already in absolute coordinates for the resized image
        # We need to scale them back to original size
        width_scale = original_size[0] / INPUT_SIZE[0]
        height_scale = original_size[1] / INPUT_SIZE[1]
        
        boxes[:, [0, 2]] *= width_scale  # x coordinates
        boxes[:, [1, 3]] *= height_scale  # y coordinates
        
        return labels, boxes, scores
    
    def visualize(self, image, labels, boxes, scores, save_path=None):
        """Draw bounding boxes on image."""
        draw = ImageDraw.Draw(image)
        
        # Try to use a better font
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()
        
        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            # Draw label
            class_name = CLASS_NAMES.get(int(label), f'class_{int(label)}')
            text = f'{class_name}: {score:.2f}'
            
            # Draw text with background
            bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle(bbox, fill='red')
            draw.text((x1, y1), text, fill='white', font=font)
        
        # Save or show
        if save_path:
            image.save(save_path)
            print(f"Saved annotated image to: {save_path}")
        
        return image
    
    def detect(self, image_path, save_path=None, threshold=CONFIDENCE_THRESHOLD):
        """Complete detection pipeline."""
        # Preprocess
        original_image, image_batch, original_size = self.preprocess(image_path)
        
        # Inference
        outputs = self.infer(image_batch)
        
        # Postprocess
        labels, boxes, scores = self.postprocess(outputs, original_size, threshold)
        
        # Visualize
        annotated_image = self.visualize(original_image, labels, boxes, scores, save_path)
        
        print(f"Detected {len(labels)} objects")
        
        return labels, boxes, scores, annotated_image


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RT-DETR ONNX Inference')
    parser.add_argument('--model', type=str, default='rtdetr_r18.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='output_annotated.jpg',
                        help='Path to save annotated image')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Confidence threshold')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RTDETRInference(args.model, use_gpu=not args.cpu)
    
    # Run detection
    labels, boxes, scores, annotated_image = detector.detect(
        args.image, 
        args.output, 
        args.threshold
    )
    
    # Print results
    print(f"\nDetection Results:")
    for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
        class_name = CLASS_NAMES.get(int(label), f'class_{int(label)}')
        print(f"  {i+1}. {class_name}: {score:.3f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")


if __name__ == "__main__":
    main()
```

---

## Step 4: Run Inference

### Single Image

```powershell
python local_inference.py --image "path\to\your\image.jpg" --output "result.jpg"
```

### With Custom Threshold

```powershell
python local_inference.py --image "test.jpg" --output "result.jpg" --threshold 0.5
```

### CPU Only (if no GPU)

```powershell
python local_inference.py --image "test.jpg" --cpu
```

---

## Quick Test Script

Create `test_onnx.py` to verify everything works:

```python
# test_onnx.py
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("rtdetr_r18.onnx")

# Print model info
print("Model inputs:")
for input in session.get_inputs():
    print(f"  {input.name}: {input.shape} ({input.type})")

print("\nModel outputs:")
for output in session.get_outputs():
    print(f"  {output.name}: {output.shape} ({output.type})")

# Test inference with dummy data
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})

print(f"\nInference test successful!")
print(f"Output shapes: {[o.shape for o in outputs]}")
```

---

## Folder Structure

```
D:\MyDeployment\
├── rtdetr_r18.onnx           # Downloaded from Kaggle
├── local_inference.py         # Main inference script
├── test_onnx.py              # Test script
├── test_images\              # Your test images
│   └── example.jpg
└── results\                  # Output folder
    └── example_annotated.jpg
```

---

## Troubleshooting

### Issue: "CUDAExecutionProvider not available"
**Solution**: GPU not detected. Either:
- Install CUDA 11.x or 12.x
- Use `--cpu` flag for CPU inference

### Issue: "DLL load failed" on Windows
**Solution**: Install Visual C++ Redistributable:
[Download here](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Issue: Slow inference
**Check**:
```python
print(session.get_providers())
# Should show: ['CUDAExecutionProvider', ...]
# If only 'CPUExecutionProvider', GPU not being used
```

---

## Performance Expectations

| Hardware | Expected FPS |
|----------|--------------|
| CPU (Intel i7) | 5-10 FPS |
| GPU (GTX 1660) | 30-50 FPS |
| GPU (RTX 3060) | 60-80 FPS |
| GPU (RTX 4090) | 100+ FPS |

---

## Next Steps

1. Export model on Kaggle
2. Download `.onnx` file
3. Install dependencies on your PC
4. Run `local_inference.py`
5. Enjoy fast inference! 🚀
