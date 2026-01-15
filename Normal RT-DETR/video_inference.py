"""
RT-DETR Video Inference Script
Process videos with RT-DETR ONNX model
Python 3.12 compatible
"""

import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import sys
import time

# Force add CUDA paths to environment for current session
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(cuda_path):
    os.environ["PATH"] += os.pathsep + cuda_path
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(cuda_path)

# Configuration
INPUT_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.4

# VisDrone class names (1-indexed as per VisDrone annotations)
CLASS_NAMES = {
    1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
    6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
}

# Colors for different classes (BGR format for OpenCV)
CLASS_COLORS = {
    1: (255, 0, 0),    # pedestrian - Blue
    2: (0, 255, 0),    # people - Green
    3: (0, 0, 255),    # bicycle - Red
    4: (255, 255, 0),  # car - Cyan
    5: (255, 0, 255),  # van - Magenta
    6: (0, 255, 255),  # truck - Yellow
    7: (128, 0, 128),  # tricycle - Purple
    8: (0, 128, 128),  # awning-tricycle - Teal
    9: (128, 128, 0),  # bus - Olive
    10: (255, 128, 0), # motor - Orange
}


class RTDETRVideoInference:
    def __init__(self, model_path, use_gpu=True):
        """Initialize ONNX Runtime session."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        active_provider = self.session.get_providers()[0]
        print(f"[+] Model loaded successfully!")
        print(f"[+] Using: {active_provider}")
        
        if use_gpu and 'CUDA' not in active_provider:
            print("[!] GPU requested but not available, falling back to CPU")
    
    def preprocess_frame(self, frame):
        """Preprocess video frame."""
        # Convert BGR (OpenCV) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(frame_rgb)
        original_size = image.size  # (width, height)
        
        # Resize to model input size
        image_resized = image.resize(INPUT_SIZE)
        
        # Convert to numpy array and normalize
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        
        # Transpose to CHW format
        image_np = image_np.transpose(2, 0, 1)
        
        # Add batch dimension
        image_batch = np.expand_dims(image_np, 0)
        
        return image_batch, original_size
    
    def infer(self, image_batch, orig_size):
        """Run inference."""
        orig_size_tensor = np.array([[orig_size[0], orig_size[1]]], dtype=np.int64)
        
        inputs = {
            'images': image_batch,
            'orig_target_sizes': orig_size_tensor
        }
        
        outputs = self.session.run(None, inputs)
        return outputs
    
    def postprocess(self, outputs, threshold=CONFIDENCE_THRESHOLD):
        """Process model outputs."""
        labels = outputs[0][0]
        boxes = outputs[1][0]
        scores = outputs[2][0]
        
        # Filter by confidence
        mask = scores > threshold
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]
        
        return labels, boxes, scores
    
    def draw_detections(self, frame, labels, boxes, scores):
        """Draw bounding boxes on frame."""
        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # Get class info
            class_id = int(label)
            class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
            color = CLASS_COLORS.get(class_id, (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            text = f'{class_name}: {score:.2f}'
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 5), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                frame, 
                text, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        return frame
    
    def process_video(self, video_path, output_path, threshold=CONFIDENCE_THRESHOLD, 
                     show_fps=True, skip_frames=0):
        """Process entire video."""
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f}s")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        print(f"\nProcessing video...")
        print(f"  Threshold: {threshold}")
        print(f"  Skip frames: {skip_frames} (process every {skip_frames + 1} frame)")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if requested
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                out.write(frame)  # Write original frame
                continue
            
            # Inference
            image_batch, original_size = self.preprocess_frame(frame)
            outputs = self.infer(image_batch, original_size)
            labels, boxes, scores = self.postprocess(outputs, threshold)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, labels, boxes, scores)
            
            # Add FPS counter
            if show_fps:
                current_time = time.time()
                elapsed = current_time - start_time
                processing_fps = processed_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(
                    annotated_frame,
                    f'Processing FPS: {processing_fps:.1f} | Detections: {len(labels)}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Write frame
            out.write(annotated_frame)
            processed_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_processing = processed_count / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
                
                print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
                      f"Processing FPS: {fps_processing:.1f} | ETA: {eta:.1f}s", end='\r')
        
        # Cleanup
        cap.release()
        out.release()
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = processed_count / total_time
        
        print(f"\n\n✓ Video processing complete!")
        print(f"  Processed {processed_count} frames in {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Output saved to: {output_path}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RT-DETR Video Inference')
    parser.add_argument('--model', type=str, default='model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video (default: input_name_annotated.mp4)')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Confidence threshold (default: 0.4)')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Skip N frames between processing (0 = process all frames)')
    parser.add_argument('--no-fps', action='store_true',
                        help='Do not show FPS counter on video')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{base_name}_annotated.mp4"
    
    print("="*60)
    print("RT-DETR R18 Video Inference")
    print("="*60)
    
    # Initialize detector
    detector = RTDETRVideoInference(args.model, use_gpu=not args.cpu)
    
    # Process video
    detector.process_video(
        args.video,
        args.output,
        args.threshold,
        show_fps=not args.no_fps,
        skip_frames=args.skip_frames
    )
    
    print("\n" + "="*60)
    print("✓ Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
