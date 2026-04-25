import cv2
import numpy as np
import os
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_FILE = os.path.join(_BASE_DIR, "yolov8n-face.pt")

_yolo_model = None
if YOLO and os.path.exists(_MODEL_FILE):
    _yolo_model = YOLO(_MODEL_FILE)
    print("[face_detector] Loaded YOLOv8 face detector.")
else:
    print("[face_detector] WARNING: YOLOv8 face model not found or ultralytics not installed.")

_prev_boxes     = []
_BOX_ALPHA      = 0.8
_CONF_THRESHOLD = 0.35   # strict — avoids false positives


def _compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
    return inter / (union + 1e-5)


def get_faces_dnn(frame, smooth=True, min_size=20):
    """
    Detect faces in *frame* using YOLOv8-face network.

    Returns a list of (x, y, w, h) tuples.
    If smooth=True, applies EMA tracking against _prev_boxes for stable boxes.
    """
    global _prev_boxes

    if _yolo_model is None:
        return _prev_boxes if smooth else []

    h, w = frame.shape[:2]
    results = _yolo_model.predict(frame, conf=_CONF_THRESHOLD, verbose=False)
    
    current_faces = []
    if len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes.xyxy:
                sX, sY, eX, eY = [int(v) for v in box.tolist()]
                sX, sY = max(0, sX), max(0, sY)
                eX, eY = min(w, eX), min(h, eY)
                fw, fh = eX - sX, eY - sY
                if fw >= min_size and fh >= min_size:
                    current_faces.append((sX, sY, fw, fh))

    if not smooth:
        return current_faces

    # Ghost-box prevention: never carry forward old boxes when nothing is visible
    if not current_faces:
        _prev_boxes = []
        return []

    # Multi-face EMA tracking
    smoothed   = []
    unmatched  = _prev_boxes.copy()
    for c in current_faces:
        best_iou, best_idx = 0, -1
        for i, p in enumerate(unmatched):
            iou = _compute_iou(c, p)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx != -1 and best_iou > 0.3:
            p = unmatched.pop(best_idx)
            s = tuple(int(_BOX_ALPHA * pi + (1 - _BOX_ALPHA) * ci)
                      for pi, ci in zip(p, c))
            smoothed.append(s)
        else:
            smoothed.append(c)

    _prev_boxes = smoothed
    return smoothed


def clear_detector_state():
    global _prev_boxes
    _prev_boxes = []
