"""
face_detector.py  — Enhanced face detector (YuNet + frame preprocessing)

Detection pipeline:
    1. CLAHE brightness normalisation
    2. Slight sharpening on low-contrast frames
    3. OpenCV FaceDetectorYN inference with confidence 0.40
    4. EMA box smoothing for stable bounding boxes
"""

import cv2
import numpy as np
import os

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_FILE = os.path.join(_BASE_DIR, "weights", "face_detection_yunet.onnx")

_yunet_model = None
if os.path.exists(_MODEL_FILE):
    # Initialize with a default size; will be set per frame dynamically
    _yunet_model = cv2.FaceDetectorYN_create(_MODEL_FILE, "", (320, 320), 0.4, 0.3, 5000)
    print("[face_detector] Loaded YuNet face detector ✓")
else:
    print("[face_detector] WARNING: YuNet model not found; detection disabled.")

_prev_boxes     = []
_BOX_ALPHA      = 0.75          # EMA weight for temporal smoothing

# ── CLAHE enhancer (reused across calls) ─────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Sharpening kernel
_SHARPEN_KERNEL = np.array([[ 0, -1,  0],
                             [-1,  5, -1],
                             [ 0, -1,  0]], dtype=np.float32)


def _enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE brightness normalisation + optional gentle sharpening.
    Converts to LAB, applies CLAHE on L channel, converts back.
    """
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Sharpen only if frame is slightly blurry (Laplacian variance heuristic)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 120:
        enhanced = cv2.filter2D(enhanced, -1, _SHARPEN_KERNEL)

    return enhanced


def _compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
    return inter / (union + 1e-5)


def get_faces_dnn(frame: np.ndarray, smooth: bool = True, min_size: int = 20):
    """
    Detect faces using YuNet.
    Returns: List of (x, y, w, h, landmarks)
    where landmarks is a list of 10 floats [x0,y0, x1,y1, x2,y2, x3,y3, x4,y4].
    """
    global _prev_boxes

    if _yunet_model is None:
        return _prev_boxes if smooth else []

    h, w = frame.shape[:2]
    
    # Preprocessing: enhance before detection
    enhanced = _enhance_frame(frame)

    _yunet_model.setInputSize((w, h))
    _, results = _yunet_model.detect(enhanced)

    current_faces = []
    if results is not None:
        for face in results:
            sX, sY, fw, fh = [int(v) for v in face[0:4]]
            landmarks = face[4:14].tolist()
            conf = face[14]
            if fw >= min_size and fh >= min_size:
                # YuNet boxes can go out of bounds sometimes
                sX, sY = max(0, sX), max(0, sY)
                if sX + fw > w: fw = w - sX
                if sY + fh > h: fh = h - sY
                current_faces.append((sX, sY, fw, fh, landmarks, conf))

    if not smooth:
        return current_faces

    # Ghost-box prevention
    if not current_faces:
        _prev_boxes = []
        return []

    # Multi-face EMA tracking (only tracking bbox, not landmarks/conf)
    smoothed  = []
    unmatched = _prev_boxes.copy()
    for c in current_faces:
        c_box = c[0:4]
        c_rest = c[4:]
        best_iou, best_idx = 0, -1
        for i, p in enumerate(unmatched):
            iou = _compute_iou(c_box, p[0:4])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx != -1 and best_iou > 0.25:
            p = unmatched.pop(best_idx)
            s_box = tuple(int(_BOX_ALPHA * pi + (1 - _BOX_ALPHA) * ci)
                      for pi, ci in zip(p[0:4], c_box))
            smoothed.append((*s_box, *c_rest))
        else:
            smoothed.append(c)

    _prev_boxes = smoothed
    return smoothed


def clear_detector_state():
    global _prev_boxes
    _prev_boxes = []


def detect_faces_multiscale(frame: np.ndarray, min_size: int = 20) -> list:
    """
    Run face detection at 1x, 1.5x and 2x upscales and merge results.
    Applies strict NMS (IoU=0.45) to eliminate duplicates.
    Returns list of (x, y, w, h, landmarks, conf) in ORIGINAL frame coordinates.
    """
    if _yunet_model is None:
        return []

    h, w = frame.shape[:2]
    all_faces: list = []

    for scale in (1.0, 1.5, 2.0):
        if scale > 1.0:
            nh = int(h * scale)
            nw = int(w * scale)
            scaled = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_CUBIC)
        else:
            scaled = frame

        faces = get_faces_dnn(scaled, smooth=False, min_size=max(10, int(min_size * scale)))

        for (x, y, fw, fh, landmarks, conf) in faces:
            ox = int(x / scale)
            oy = int(y / scale)
            ow = max(min_size, int(fw / scale))
            oh = max(min_size, int(fh / scale))
            orig_lm = [lm / scale for lm in landmarks]
            all_faces.append((ox, oy, ow, oh, orig_lm, conf))

    # Strict NMS with IoU 0.45
    all_faces.sort(key=lambda f: f[5], reverse=True)
    merged = []
    
    for face in all_faces:
        keep = True
        for kept_face in merged:
            if _compute_iou(face[0:4], kept_face[0:4]) > 0.45:
                keep = False
                break
        if keep:
            merged.append(face)

    return merged

