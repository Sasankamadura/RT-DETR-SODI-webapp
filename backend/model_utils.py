import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# VisDrone class names
CLASS_NAMES_1_INDEXED = {
    1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
    6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
}

CLASS_NAMES_0_INDEXED = {
    0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van',
    5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'
}

class ModelHandler:
    def __init__(self, model_path, indexing_type="0-indexed"):
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = CLASS_NAMES_0_INDEXED if indexing_type == "0-indexed" else CLASS_NAMES_1_INDEXED

    def preprocess(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size
        image_resized = image.resize((640, 640))
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        image_np = image_np.transpose(2, 0, 1)
        image_batch = np.expand_dims(image_np, 0)
        return image, image_batch, original_size

    def predict(self, image_bytes, conf_threshold=0.4):
        original_image, image_batch, original_size = self.preprocess(image_bytes)
        
        orig_size_tensor = np.array([[original_size[0], original_size[1]]], dtype=np.int64)
        inputs = {self.input_name: image_batch, 'orig_target_sizes': orig_size_tensor}
        
        outputs = self.session.run(None, inputs)
        
        labels = outputs[0][0]
        boxes = outputs[1][0]
        scores = outputs[2][0]
        
        mask = scores > conf_threshold
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]
        
        # Draw annotations
        draw = ImageDraw.Draw(original_image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        detections = []
        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box.tolist()
            class_name = self.class_names.get(int(label), str(int(label)))
            color = "#00FF00"  # Green
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"{class_name} {score:.2f}"
            
            # Text background
            bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=color)
            draw.text((x1, y1), text, fill="black", font=font)

            detections.append({
                "box": [x1, y1, x2, y2],
                "label_id": int(label),
                "label": class_name,
                "score": float(score)
            })
            
        return original_image, detections
