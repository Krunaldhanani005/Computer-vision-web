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
    'cell phone': 0.45,   # restored sensitivity for real phones
    'remote':     0.01,   # ultra-low threshold: YOLO's hidden remote predictions will suppress phones
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
    'cell phone': 3,   # stricter — remotes cause 1-2 frame flickers
    'remote':     2,
}

# ══════════════════════════════════════════════════════════════════════════════
#  STATE  (module-level, reset automatically as objects appear/disappear)
# ══════════════════════════════════════════════════════════════════════════════
_TRACKS: dict[int, dict] = {}
_NEXT_TRACK_ID = 1


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

def _track_objects(detections: list) -> list:
    global _NEXT_TRACK_ID, _TRACKS
    
    matched_tracks = set()
    results = []
    
    for det in detections:
        cls = det['class_name']
        box = det['bbox']
        conf = det['confidence']
        
        best_iou = 0.30
        best_id = None
        for tid, track in _TRACKS.items():
            if track['class'] != cls:
                continue
            iou = _iou(box, track['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_id = tid
                
        if best_id is not None:
            track = _TRACKS[best_id]
            matched_tracks.add(best_id)
            track['hits'] += 1
            track['age'] = 0
            
            old_box = track['bbox']
            
            cx_old = (old_box[0] + old_box[2]) / 2
            cy_old = (old_box[1] + old_box[3]) / 2
            cx_new = (box[0] + box[2]) / 2
            cy_new = (box[1] + box[3]) / 2
            dist = (cx_old - cx_new)**2 + (cy_old - cy_new)**2
            
            if cls == 'chair':
                # Chair stabilization logic
                if conf < 0.65 and track['hits'] > 5:
                    new_box = old_box
                elif dist < 64 or best_iou > 0.85: # less than 8px move -> lock box
                    new_box = old_box
                else:
                    alpha = 0.15
                    new_box = tuple(int(old_box[i] * (1-alpha) + box[i] * alpha) for i in range(4))
            else:
                alpha = 0.40
                new_box = tuple(int(old_box[i] * (1-alpha) + box[i] * alpha) for i in range(4))
                
            track['bbox'] = new_box
            track['last_conf'] = conf
            
            results.append({
                'track_id': best_id,
                'class_name': cls,
                'confidence': conf,
                'bbox': new_box
            })
        else:
            tid = _NEXT_TRACK_ID
            _NEXT_TRACK_ID += 1
            _TRACKS[tid] = {
                'class': cls,
                'bbox': box,
                'age': 0,
                'hits': 1,
                'last_conf': conf
            }
            matched_tracks.add(tid)
            results.append({
                'track_id': tid,
                'class_name': cls,
                'confidence': conf,
                'bbox': box
            })
            
    # Age out and persistence
    for tid in list(_TRACKS.keys()):
        if tid not in matched_tracks:
            _TRACKS[tid]['age'] += 1
            persistence = 10 if _TRACKS[tid]['class'] == 'chair' else 3
            if _TRACKS[tid]['age'] > persistence:
                del _TRACKS[tid]
            else:
                results.append({
                    'track_id': tid,
                    'class_name': _TRACKS[tid]['class'],
                    'confidence': _TRACKS[tid].get('last_conf', 0.0),
                    'bbox': _TRACKS[tid]['bbox']
                })
                
    confirmed = []
    for r in results:
        cls = r['class_name']
        if cls == 'remote':
            continue
        req_hits = _CONFIRM_FRAMES.get(cls, 3)
        if _TRACKS[r['track_id']]['hits'] >= req_hits:
            confirmed.append({
                'class_name': cls,
                'confidence': r['confidence'],
                'bbox': r['bbox']
            })
            
    return confirmed


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
        # Real phones are rarely narrower than 0.42 aspect ratio.
        # Elongated AC remotes are typically 0.25–0.38.
        if aspect > 2.0 or aspect < 0.40:
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
    # conf=0.01 exposes weak predictions so our cross-class NMS can use them
    results = model(processed, imgsz=704, conf=0.01, iou=0.45, verbose=False)

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
                
            # ── AC Remote Visual Heuristic ────────────────────────────────────
            # To strictly prevent white AC remotes from being detected as phones:
            if cls_name == 'cell phone' and h > 60 and w > 30:
                aspect = w / (h + 1e-5)
                # AC remotes are tall (aspect < 0.65)
                if aspect < 0.65:
                    # Take center 50% width to avoid fingers
                    cx1, cx2 = x1 + int(w * 0.25), x2 - int(w * 0.25)
                    if cx2 > cx1:
                        crop = frame[y1:y2, cx1:cx2]
                        if crop.size > 0:
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            split_y = int(h * 0.35) # top 35% is screen
                            if 0 < split_y < h:
                                top_mean = np.mean(gray[:split_y, :])
                                bot_mean = np.mean(gray[split_y:, :])
                                # AC remotes: bottom is bright white plastic, top is dark screen
                                if bot_mean > 130 and top_mean < 115 and (bot_mean - top_mean) > 40:
                                    continue # Rejected! It matches the visual profile of an AC remote.

            seen_classes.add(cls_name)
            raw_candidates.append({
                'class_name': cls_name,
                'confidence': conf,
                'bbox':       (x1, y1, x2, y2),
            })

    # ── 4. Cross-class conflict resolution ────────────────────────────────────
    resolved = _cross_class_nms(raw_candidates)

    # ── 5. Temporal tracking and confirmation ─────────────────────────────────
    confirmed_dets = _track_objects(resolved)

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
