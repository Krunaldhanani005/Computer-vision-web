"""
face_detector.py  — Enhanced face detector (YuNet + adaptive preprocessing)

Detection pipeline:
    1. Adaptive CLAHE — only when frame is dark (mean-L < 110) or low-contrast (std-L < 35)
    2. Gentle sharpening — only when Laplacian variance < 180
    3. YuNet inference at score_threshold=0.30 (↓ from 0.40) for better recall on
       side faces, far faces, and faces with partial occlusion
    4. Multi-scale pass (1.0×, 1.5×, 2.0×) with scale-cap at 1280 px to prevent lag
    5. Strict NMS — IoU 0.35 (↓ from 0.45) eliminates cross-scale duplicates
    6. EMA box smoothing α=0.55 (↓ from 0.75) — tracks moving faces without lag

Key parameter rationale
    score_threshold  0.30  catches side angles up to ~60°, medium-distance faces
    nms_threshold    0.25  tighter internal YuNet NMS — fewer raw duplicates
    _BOX_ALPHA       0.55  half-weight on previous pos → responsive but not jittery
    _EMA_IOU_MATCH   0.15  allows matching boxes that moved more between frames
    NMS_IOU          0.35  removes cross-scale dupes without merging nearby faces
"""

import cv2
import numpy as np
import os

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_FILE = os.path.join(_BASE_DIR, "weights", "face_detection_yunet.onnx")

_SCORE_THRESHOLD = 0.38   # balanced: recalls far/medium faces; downstream gates filter FPs
_NMS_THRESHOLD   = 0.25   # was 0.30 — tighter internal NMS
_TOP_K           = 5000

_yunet_model = None
if os.path.exists(_MODEL_FILE):
    _yunet_model = cv2.FaceDetectorYN_create(
        _MODEL_FILE, "", (320, 320),
        score_threshold=_SCORE_THRESHOLD,
        nms_threshold=_NMS_THRESHOLD,
        top_k=_TOP_K,
    )
    print("[face_detector] Loaded YuNet face detector ✓  "
          f"(score≥{_SCORE_THRESHOLD}, nms={_NMS_THRESHOLD})")
else:
    print("[face_detector] WARNING: YuNet model not found; detection disabled.")

_prev_boxes: list = []
_BOX_ALPHA       = 0.55   # EMA weight on previous position (was 0.75)
_EMA_IOU_MATCH   = 0.15   # min IoU to link new box → existing track (was 0.25)
_NMS_IOU         = 0.35   # multi-scale NMS IoU threshold (was 0.45)

# CLAHE — reused across calls
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

# Sharpening kernel (gentle — avoids edge ringing on noisy CCTV)
_SHARPEN_KERNEL = np.array(
    [[ 0, -0.5,  0],
     [-0.5, 3.0, -0.5],
     [ 0, -0.5,  0]], dtype=np.float32
)


# ── Frame preprocessing ───────────────────────────────────────────────────────

def _enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Adaptive enhancement — only processes what needs processing.
    Avoids degrading already-good frames (bright, high-contrast).

    CLAHE applied when: mean-L < 110 (dark) OR std-L < 35 (low contrast)
    Sharpening applied when: Laplacian variance < 180 (blurry)
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    mean_l = float(l.mean())
    std_l  = float(l.std())

    if mean_l < 110 or std_l < 35:
        l_eq     = _clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    else:
        enhanced = frame

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 180:   # was 120
        enhanced = cv2.filter2D(enhanced, -1, _SHARPEN_KERNEL)

    return enhanced


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _compute_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
    return inter / (union + 1e-5)


def _validate_landmarks(x: int, y: int, w: int, h: int, lm: list) -> bool:
    """
    Sanity-check 5-point landmarks.
    Requires ≥ 3 of 5 points to lie within a generous ±60% padded face box.
    Rejects wildly wrong landmark sets from false positives.
    """
    if lm is None or len(lm) < 10:
        return False
    px, py   = w * 0.60, h * 0.60
    x_lo, x_hi = x - px, x + w + px
    y_lo, y_hi = y - py, y + h + py
    valid = sum(
        1 for i in range(5)
        if x_lo <= lm[2 * i] <= x_hi and y_lo <= lm[2 * i + 1] <= y_hi
    )
    return valid >= 3


# ── Core detection ────────────────────────────────────────────────────────────

def get_faces_dnn(frame: np.ndarray, smooth: bool = True, min_size: int = 20):
    """
    Detect faces using YuNet with adaptive preprocessing.

    Returns: list of (x, y, w, h, landmarks_or_None, conf)
        landmarks = list of 10 floats [x0,y0 … x4,y4] or None if validation fails
    """
    global _prev_boxes

    if _yunet_model is None:
        return _prev_boxes if smooth else []

    h, w = frame.shape[:2]
    enhanced = _enhance_frame(frame)

    _yunet_model.setInputSize((w, h))
    _, results = _yunet_model.detect(enhanced)

    current_faces = []
    if results is not None:
        for face in results:
            sX = int(face[0]);  sY = int(face[1])
            fw = int(face[2]);  fh = int(face[3])
            landmarks = face[4:14].tolist()
            conf      = float(face[14])

            if fw < min_size or fh < min_size:
                continue

            # Clamp to frame bounds
            sX = max(0, sX);  sY = max(0, sY)
            fw = min(fw, w - sX)
            fh = min(fh, h - sY)
            if fw < min_size or fh < min_size:
                continue

            # Validate landmarks — mark as None if clearly wrong
            lm = landmarks if _validate_landmarks(sX, sY, fw, fh, landmarks) else None

            current_faces.append((sX, sY, fw, fh, lm, conf))

    if not smooth:
        return current_faces

    # Ghost-box prevention
    if not current_faces:
        _prev_boxes = []
        return []

    # EMA tracking — match new detections to previous positions
    smoothed  = []
    unmatched = list(_prev_boxes)

    for c in current_faces:
        c_box  = c[0:4]
        c_rest = c[4:]          # (landmarks, conf)
        best_iou, best_idx = 0.0, -1

        for i, p in enumerate(unmatched):
            iou = _compute_iou(c_box, p[0:4])
            if iou > best_iou:
                best_iou, best_idx = iou, i

        if best_idx != -1 and best_iou > _EMA_IOU_MATCH:
            p     = unmatched.pop(best_idx)
            s_box = tuple(
                int(_BOX_ALPHA * pi + (1 - _BOX_ALPHA) * ci)
                for pi, ci in zip(p[0:4], c_box)
            )
            smoothed.append((*s_box, *c_rest))
        else:
            smoothed.append(c)  # new detection — no prior to blend with

    _prev_boxes = smoothed
    return smoothed


def clear_detector_state():
    global _prev_boxes
    _prev_boxes = []


# ── Multi-scale detection ─────────────────────────────────────────────────────

def detect_faces_multiscale(frame: np.ndarray, min_size: int = 20) -> list:
    """
    Three-scale face detection for robust recall across distances.

    Scales 1.0×, 1.5×, 2.0× cover:
        1.0× — close / large faces (webcam, near CCTV)
        1.5× — medium-distance faces (typical walking person crop)
        2.0× — small / far faces (CCTV crowd scenes)

    Scale cap: max dimension capped at 1280 px to prevent lag on large crops.
    Cross-scale duplicates removed by NMS (IoU ≥ 0.35, highest-conf wins).

    Returns: list of (x, y, w, h, landmarks_or_None, conf) in ORIGINAL coords.
    """
    if _yunet_model is None:
        return []

    h, w       = frame.shape[:2]
    all_faces: list = []

    for scale in (1.0, 1.5, 2.0):
        actual_scale = scale
        if scale > 1.0:
            nw = int(w * scale)
            nh = int(h * scale)
            # Cap so no dimension exceeds 1280 px — keeps per-frame time bounded
            if max(nw, nh) > 1280:
                actual_scale = 1280 / max(w, h)
                nw = int(w * actual_scale)
                nh = int(h * actual_scale)
            scaled = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            scaled = frame

        scale_min = max(10, int(min_size * actual_scale))
        faces     = get_faces_dnn(scaled, smooth=False, min_size=scale_min)

        for face in faces:
            x   = face[0];  y   = face[1]
            fw  = face[2];  fh  = face[3]
            lm  = face[4];  conf = face[5]

            ox = int(x  / actual_scale)
            oy = int(y  / actual_scale)
            ow = max(min_size, int(fw / actual_scale))
            oh = max(min_size, int(fh / actual_scale))
            orig_lm = ([v / actual_scale for v in lm] if lm is not None else None)

            all_faces.append((ox, oy, ow, oh, orig_lm, conf))

    if not all_faces:
        return []

    # Confidence-sorted NMS — keeps best landmark set when boxes overlap
    all_faces.sort(key=lambda f: f[5], reverse=True)
    merged: list = []
    for face in all_faces:
        keep = True
        for kept in merged:
            if _compute_iou(face[0:4], kept[0:4]) > _NMS_IOU:
                keep = False
                break
        if keep:
            merged.append(face)

    return merged
