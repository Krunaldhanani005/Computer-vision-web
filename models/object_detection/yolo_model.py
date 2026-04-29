import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Load model once
model = YOLO("yolov8s.pt")

_ALLOWED = {'bottle', 'chair', 'clock', 'cell phone'}
DISPLAY_MAP = {
    'bottle': 'Bottle',
    'chair': 'Chair',
    'clock': 'Watch',
    'cell phone': 'Mobile Phone'
}

BOX_COLORS = {
    'bottle':     (255, 165, 0),    # Orange
    'chair':      (0, 200, 100),    # Green
    'clock':      (0, 200, 255),    # Yellow/Amber
    'cell phone': (255, 100, 255)   # Pink
}
DEFAULT_COLOR = (200, 200, 200)

box_history = {}

def smooth_box(cls, box):
    if cls not in box_history:
        box_history[cls] = deque(maxlen=5)
    
    box_history[cls].append(box)
    
    avg = [int(sum(x)/len(x)) for x in zip(*box_history[cls])]
    return tuple(avg)

def detect_objects(frame):
    # Use imgsz=800 to zoom in on smaller objects natively
    # Base conf is lowered to 0.2 to allow bottle and watch detection
    results = model(frame, imgsz=700, conf=0.2, verbose=False)
    detections = []
    current_classes = set()
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Class-specific control to balance detection
            if cls_name == 'bottle' and conf < 0.30: # Raised to stop blind detection
                continue
            elif cls_name == 'chair' and conf < 0.6: # Stop blind detection
                continue
            elif cls_name == 'clock' and conf < 0.2:
                continue
            elif cls_name == 'cell phone' and conf < 0.45:
                continue
            elif cls_name not in _ALLOWED:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            
            if cls_name == 'clock':
                min_size = 15
            else:
                min_size = 25
                
            # Size filter
            if w < min_size or h < min_size:
                continue
            
            # Smooth box coordinates
            x1, y1, x2, y2 = smooth_box(cls_name, (x1, y1, x2, y2))
            current_classes.add(cls_name)
            # Add detection immediately to make it responsive
            detections.append({
                'class_name': cls_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })
                    
    # Clear history for objects that disappeared
    for cls in list(box_history.keys()):
        if cls not in current_classes:
            box_history.pop(cls)
            
    return detections

def draw_detections(frame, detections):
    for det in detections:
        cls_name = det['class_name']
        conf = det['confidence']
        x1, y1, x2, y2 = det['bbox']
        
        display_label = DISPLAY_MAP.get(cls_name, cls_name)
        label_text = f"{display_label} ({int(conf * 100)}%)"
        
        # Assign unique color per class
        color = BOX_COLORS.get(cls_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw background for text
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 3, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
