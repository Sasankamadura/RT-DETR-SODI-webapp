"""
RT-DETR ONNX Local Inference Script
Minimal deployment script for running RT-DETR R18 on your local PC
Python 3.12 compatible
"""

import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# Configuration
INPUT_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.4

# VisDrone class names (1-indexed as per VisDrone annotations)
CLASS_NAMES = {
    1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
    6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
}

class RTDETRInference:
    def __init__(self, model_path, use_gpu=True):
        """Initialize ONNX Runtime session."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        active_provider = self.session.get_providers()[0]
        
        print(f"✓ Model loaded successfully!")
        print(f"✓ Using: {active_provider}")
        
        if use_gpu and 'CUDA' not in active_provider:
            print("⚠  GPU requested but not available, falling back to CPU")
    
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
    
    def infer(self, image_batch, orig_size):
        """Run inference."""
        # Model expects two inputs: images and orig_target_sizes
        orig_size_tensor = np.array([[orig_size[0], orig_size[1]]], dtype=np.int64)
        
        inputs = {
            'images': image_batch,
            'orig_target_sizes': orig_size_tensor
        }
        
        outputs = self.session.run(None, inputs)
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
        
        # Boxes are already in original image coordinates (postprocessor included in ONNX)
        # No rescaling needed
        
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
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Draw label
            class_name = CLASS_NAMES.get(int(label), f'class_{int(label)}')
            text = f'{class_name}: {score:.2f}'
            
            # Draw text with background
            text_bbox = draw.textbbox((x1, y1-20), text, font=font)
            draw.rectangle(text_bbox, fill='red')
            draw.text((x1, y1-20), text, fill='white', font=font)
        
        # Save or show
        if save_path:
            image.save(save_path)
            print(f"✓ Saved annotated image to: {save_path}")
        
        return image
    
    def detect(self, image_path, save_path=None, threshold=CONFIDENCE_THRESHOLD):
        """Complete detection pipeline."""
        print(f"\nProcessing: {image_path}")
        
        # Preprocess
        original_image, image_batch, original_size = self.preprocess(image_path)
        
        # Inference
        outputs = self.infer(image_batch, original_size)
        
        # Postprocess
        labels, boxes, scores = self.postprocess(outputs, original_size, threshold)
        
        # Visualize
        annotated_image = self.visualize(original_image, labels, boxes, scores, save_path)
        
        print(f"✓ Detected {len(labels)} objects (threshold={threshold})")
        
        return labels, boxes, scores, annotated_image


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RT-DETR ONNX Inference')
    parser.add_argument('--model', type=str, default='model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save annotated image (default: input_name_annotated.jpg)')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Confidence threshold (default: 0.4)')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        print(f"Please export your model first using export_onnx.py on Kaggle")
        sys.exit(1)
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = f"{base_name}_annotated.jpg"
    
    print("="*60)
    print("RT-DETR R18 Local Inference")
    print("="*60)
    
    # Initialize detector
    detector = RTDETRInference(args.model, use_gpu=not args.cpu)
    
    # Run detection
    labels, boxes, scores, annotated_image = detector.detect(
        args.image, 
        args.output, 
        args.threshold
    )
    
    # Print detailed results
    if len(labels) > 0:
        print(f"\nDetection Results:")
        print(f"{'#':<4} {'Class':<20} {'Confidence':<12} {'BBox (x1,y1,x2,y2)'}")
        print("-"*60)
        
        for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
            class_name = CLASS_NAMES.get(int(label), f'class_{int(label)}')
            bbox_str = f"[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"
            print(f"{i+1:<4} {class_name:<20} {score:.3f} ({score*100:.1f}%) {bbox_str}")
    else:
        print(f"\n⚠  No objects detected with confidence > {args.threshold}")
        print(f"   Try lowering the threshold with --threshold 0.3")
    
    print("\n" + "="*60)
    print("✓ Inference complete!")
    print("="*60)


if __name__ == "__main__":
    main()
