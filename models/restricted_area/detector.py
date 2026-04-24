"""
models/restricted_area/detector.py
────────────────────────────────────
YOLO-based person detection (class 0 only).
Runs on a shared yolov8n model, lightweight and fast.
"""

import os
from ultralytics import YOLO

_model = YOLO("yolov8n.pt")   # nano — fast, sufficient for person detection
_PERSON_CLASS = 0
_CONF_THRESHOLD = 0.50


def get_persons(frame) -> list[tuple[int, int, int, int]]:
    """
    Run YOLO on *frame* and return bounding boxes of detected persons.

    Returns:
        list of (x1, y1, x2, y2) integer tuples
    """
    try:
        results = _model(frame, imgsz=480, conf=_CONF_THRESHOLD,
                         classes=[_PERSON_CLASS], verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != _PERSON_CLASS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w < 30 or h < 60:          # filter tiny detections
                    continue
                boxes.append((x1, y1, x2, y2))
        return boxes
    except Exception as e:
        print(f"[restricted_area.detector] Error: {e}")
        return []
