import cv2
import numpy as np
import os

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROTOTXT   = os.path.join(_BASE_DIR, "deploy.prototxt")
_MODEL_FILE = os.path.join(_BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

_net = None
if os.path.exists(_PROTOTXT) and os.path.exists(_MODEL_FILE):
    _net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _MODEL_FILE)
else:
    print("[face_detector] WARNING: DNN model files not found. Face detection will fail.")

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
    Detect faces in *frame* using the Caffe SSD network.

    Returns a list of (x, y, w, h) tuples.
    If smooth=True, applies EMA tracking against _prev_boxes for stable boxes.
    """
    global _prev_boxes

    if _net is None:
        return _prev_boxes if smooth else []

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    _net.setInput(blob)
    detections = _net.forward()

    current_faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > _CONF_THRESHOLD:
            box        = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            sX, sY, eX, eY = box.astype("int")
            sX, sY     = max(0, sX), max(0, sY)
            eX, eY     = min(w, eX), min(h, eY)
            fw, fh     = eX - sX, eY - sY
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
