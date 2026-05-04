"""
yolo_model.py — Object detection pipeline (YOLOv8s)

Detection classes:  bottle, chair, clock (Watch), cell phone (Mobile Phone)

Pipeline stages:
  1. Frame preprocessing  (CLAHE + mild denoise + sharpen)
  2. YOLO inference        (class-specific confidence thresholds)
  3. Shape / size gates    (aspect ratio, min/max area)
  4. Cross-class NMS       (IoU-based overlap conflict resolution)
  5. Temporal confirmation (N-frame streak required before emission)
  6. Box smoothing         (EMA for stable drawing)
"""

import cv2
import os
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "weights", "yolov8s.pt"))

_ALLOWED = {'bottle', 'chair', 'clock', 'cell phone', 'remote'}

DISPLAY_MAP = {
    'bottle':     'Bottle',
    'chair':      'Chair',
    'clock':      'Watch',
    'cell phone': 'Mobile Phone',
}

BOX_COLORS = {
    'bottle':     (255, 165, 0),    # Orange
    'chair':      (0, 200, 100),    # Green
    'clock':      (0, 200, 255),    # Yellow/Amber
    'cell phone': (255, 100, 255),  # Pink
}
DEFAULT_COLOR = (200, 200, 200)

# ── Per-class confidence thresholds ───────────────────────────────────────────
CLASS_CONF = {
    'bottle':     0.35,
    'chair':      0.55,
    'clock':      0.25,
    'cell phone': 0.52,   # high to reject remote-control cross-talk
    'remote':     0.42,   # we detect remotes just to suppress phone conflicts
}

# ── Per-class minimum box size (pixels) ───────────────────────────────────────
CLASS_MIN_SIZE = {
    'bottle':     25,
    'chair':      40,
    'clock':      15,
    'cell phone': 40,
    'remote':     15,
}

# ── Temporal confirmation ─────────────────────────────────────────────────────
# Each class must be seen for N consecutive frames before it's emitted.
_CONFIRM_FRAMES = {
    'bottle':     2,
    'chair':      3,
    'clock':      2,
    'cell phone': 4,   # stricter — remotes cause 1-2 frame flickers
    'remote':     2,
}

# ── Box smoothing ─────────────────────────────────────────────────────────────
_BOX_HISTORY_LEN = 6       # frames to average over
_BOX_ALPHA       = 0.45    # EMA weight for new box (higher = less smooth)


# ══════════════════════════════════════════════════════════════════════════════
#  STATE  (module-level, reset automatically as objects appear/disappear)
# ══════════════════════════════════════════════════════════════════════════════
_streak: dict[str, int]        = defaultdict(int)     # consecutive-frame counters
_confirmed: dict[str, bool]    = defaultdict(bool)    # True once streak >= threshold
_smooth_boxes: dict[str, list] = {}                   # EMA-smoothed box per class
_box_ring: dict[str, deque]    = {}                   # raw box history per class


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Fast preprocessing to improve detection in variable lighting.
    Budget: <5 ms total on a typical frame.
    Returns a COPY — the original frame is NOT modified.
    """
    out = frame.copy()

    # 1. CLAHE on the L channel (contrast enhancement, ~1 ms)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 2. Fast denoise via bilateral filter (~2 ms, replaces the 150 ms
    #    fastNlMeansDenoisingColored that was blocking the stream)
    out = cv2.bilateralFilter(out, d=5, sigmaColor=35, sigmaSpace=35)

    # 3. Light sharpen via unsharp mask (~1 ms)
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.5)
    out  = cv2.addWeighted(out, 1.25, blur, -0.25, 0)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _iou(a, b) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _smooth_box(cls: str, box: tuple) -> tuple:
    """EMA-based box smoothing per class — removes jitter across frames."""
    if cls not in _smooth_boxes:
        _smooth_boxes[cls] = list(box)
        _box_ring[cls]     = deque(maxlen=_BOX_HISTORY_LEN)

    _box_ring[cls].append(box)

    # EMA update
    s = _smooth_boxes[cls]
    a = _BOX_ALPHA
    for i in range(4):
        s[i] = int(s[i] * (1 - a) + box[i] * a)
    _smooth_boxes[cls] = s
    return tuple(s)


# ══════════════════════════════════════════════════════════════════════════════
#  SHAPE / SIZE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def _validate_shape(cls: str, x1, y1, x2, y2, frame_h, frame_w) -> bool:
    """Return True if the detection passes class-specific shape / area gates."""
    w = x2 - x1
    h = y2 - y1
    area = w * h
    aspect = w / (h + 1e-5)

    # ── Common: reject absurdly large boxes (>40 % of frame) ──────────────
    if area > 0.40 * frame_h * frame_w:
        return False

    # ── Cell phone ────────────────────────────────────────────────────────
    if cls == 'cell phone':
        # Phones are roughly portrait (aspect 0.35–1.8)
        if aspect > 2.0 or aspect < 0.28:
            return False
        # Must have minimum pixel area (remotes are smaller on camera)
        if w < 35 or h < 50:
            return False
        # Reject if located in top-third of frame and very small — likely
        # misclassified face/head region
        if y1 < frame_h * 0.25 and area < 4000:
            return False
        return True

    # ── Bottle ────────────────────────────────────────────────────────────
    if cls == 'bottle':
        # Bottles are tall & narrow (aspect 0.15–1.0)
        if aspect > 1.5:
            return False
        # Reject bottle detections sitting on top-quarter of frame at small
        # sizes — usually a face being misclassified as a bottle
        cy = (y1 + y2) / 2
        if cy < frame_h * 0.30 and area < 3500:
            return False
        return True

    # ── Remote ────────────────────────────────────────────────────────────
    if cls == 'remote':
        # remotes are typically small/medium and squat
        return True

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-CLASS NMS  (conflict resolution)
# ══════════════════════════════════════════════════════════════════════════════
_CONFLICT_IOU_THRESH = 0.30   # if two boxes overlap by >30 %, resolve conflict

# Priority: when two classes overlap, which one wins?
# Lower number = higher priority (more likely to be the real object).
_CLASS_PRIORITY = {
    'remote':     1,   # if remote and phone overlap → keep remote
    'chair':      2,
    'clock':      3,
    'bottle':     4,
    'cell phone': 5,   # lowest priority — most prone to false positives
}

def _cross_class_nms(candidates: list) -> list:
    """
    Suppress lower-priority detections that overlap with higher-priority ones.
    Also: if 'remote' and 'cell phone' overlap, always keep remote (the phone
    detection is almost certainly a misclassification of the remote).
    """
    if len(candidates) <= 1:
        return candidates

    # Sort by priority (lower = keep first)
    candidates.sort(key=lambda d: _CLASS_PRIORITY.get(d['class_name'], 99))

    keep = []
    for det in candidates:
        suppressed = False
        for kept in keep:
            if _iou(det['bbox'], kept['bbox']) > _CONFLICT_IOU_THRESH:
                # Special rule: remote always beats cell phone
                if det['class_name'] == 'cell phone' and kept['class_name'] == 'remote':
                    suppressed = True
                    break
                # General rule: lower priority is suppressed
                if _CLASS_PRIORITY.get(det['class_name'], 99) > _CLASS_PRIORITY.get(kept['class_name'], 99):
                    suppressed = True
                    break
        if not suppressed:
            keep.append(det)

    return keep


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL CONFIRMATION
# ══════════════════════════════════════════════════════════════════════════════
def _update_streaks(seen_classes: set):
    """
    Update per-class streak counters.
    Classes present this frame get their streak incremented.
    Classes absent are reset to 0.
    """
    all_tracked = set(_streak.keys()) | seen_classes
    for cls in all_tracked:
        if cls in seen_classes:
            _streak[cls] += 1
            threshold = _CONFIRM_FRAMES.get(cls, 3)
            if _streak[cls] >= threshold:
                _confirmed[cls] = True
        else:
            _streak[cls]    = 0
            _confirmed[cls] = False


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DETECT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_objects(frame: np.ndarray) -> list:
    """
    Full detection pipeline:
      preprocess → infer → filter → shape-validate → cross-NMS → temporal → smooth
    """
    frame_h, frame_w = frame.shape[:2]

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    processed = _preprocess(frame)

    # ── 2. YOLO inference ─────────────────────────────────────────────────────
    results = model(processed, imgsz=704, conf=0.20, iou=0.45, verbose=False)

    # ── 3. Class-specific threshold + size filter ─────────────────────────────
    raw_candidates = []
    seen_classes   = set()

    for r in results:
        for box in r.boxes:
            cls_id   = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf     = float(box.conf[0])

            if cls_name not in _ALLOWED:
                continue

            # Class-specific confidence gate
            if conf < CLASS_CONF.get(cls_name, 0.40):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Minimum size gate
            min_sz = CLASS_MIN_SIZE.get(cls_name, 25)
            if w < min_sz or h < min_sz:
                continue

            # Shape / spatial validation
            if not _validate_shape(cls_name, x1, y1, x2, y2, frame_h, frame_w):
                continue

            seen_classes.add(cls_name)
            raw_candidates.append({
                'class_name': cls_name,
                'confidence': conf,
                'bbox':       (x1, y1, x2, y2),
            })

    # ── 4. Cross-class conflict resolution ────────────────────────────────────
    resolved = _cross_class_nms(raw_candidates)

    # ── 5. Temporal confirmation ──────────────────────────────────────────────
    _update_streaks(seen_classes)

    confirmed_dets = []
    for det in resolved:
        cls = det['class_name']
        # 'remote' is used only for conflict suppression — never displayed
        if cls == 'remote':
            continue
        if _confirmed.get(cls, False):
            # Smooth box coordinates
            sx1, sy1, sx2, sy2 = _smooth_box(cls, det['bbox'])
            confirmed_dets.append({
                'class_name': cls,
                'confidence': det['confidence'],
                'bbox':       (sx1, sy1, sx2, sy2),
            })

    # ── 6. Clear smoothing state for classes that disappeared ─────────────────
    for cls in list(_smooth_boxes.keys()):
        if cls not in seen_classes:
            _smooth_boxes.pop(cls, None)
            _box_ring.pop(cls, None)

    return confirmed_dets


# ══════════════════════════════════════════════════════════════════════════════
#  DRAW
# ══════════════════════════════════════════════════════════════════════════════
def draw_detections(frame: np.ndarray, detections: list):
    for det in detections:
        cls_name = det['class_name']
        conf     = det['confidence']
        x1, y1, x2, y2 = det['bbox']

        display_label = DISPLAY_MAP.get(cls_name, cls_name)
        label_text    = f"{display_label} ({int(conf * 100)}%)"

        color = BOX_COLORS.get(cls_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 3, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
