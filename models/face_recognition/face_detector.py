"""
face_detector.py  — SCRFD-10G face detector

Detection pipeline:
    1. Adaptive CLAHE — throttled every _ENHANCE_EVERY frames
    2. Gentle sharpening — only when Laplacian variance < 180
    3. SCRFD inference at score_threshold=0.30
         3-stride anchor decode (strides 8 / 16 / 32, 2 anchors per cell)
         Landmark decode (5 points → 10 floats, ArcFace order)
    4. Partial-face filter — rejects boxes where > 30% area falls outside frame edge
    5. Aspect-ratio filter — 0.5 – 1.8  (tighter than before to kill FP on walls/plants)
    6. Landmark sanity check — requires >= 4/5 points inside ±25%-padded face box
    7. Confidence gate — at least 0.35 after decoding
    8. Strong NMS at IoU=0.45 — eliminates duplicate cross-stride boxes
    9. Sharpness gate — face crop Laplacian var >= 10.0

Return format:
    list of (x, y, w, h, landmarks_or_None, conf)
"""

import cv2
import numpy as np
import os
import onnxruntime as ort

# ── Model paths ───────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_CANDIDATES = [
    os.path.join(_BASE_DIR, "weights", "det_10g.onnx"),
    os.path.join(_BASE_DIR, "weights", "scrfd_10g_bnkps.onnx"),
]
_MODEL_FILE = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)), None)

# ── Detection hyper-parameters ────────────────────────────────────────────────
_SCORE_THRESHOLD  = 0.35   # raised from 0.30 → fewer background FPs
_POST_NMS_THRESH  = 0.40   # second confidence gate after NMS (kills weak survivors)
_NMS_IOU          = 0.45   # stronger NMS (was 0.40) — removes more duplicate boxes
_INPUT_SIZE       = 640    # fastest inference; upscale pass handles far faces
_STRIDES          = [8, 16, 32]
_NUM_ANCHORS      = 2
_MIN_SHARPNESS    = 12.0   # Laplacian variance threshold for face crop quality

# ── Enhancement throttle ──────────────────────────────────────────────────────
_ENHANCE_EVERY   = 3        # run CLAHE+sharpen every N frames, reuse cache otherwise
_enhance_counter  = 0
_last_enhanced: np.ndarray | None = None

# ── EMA smoothing ─────────────────────────────────────────────────────────────
_BOX_ALPHA     = 0.40
_EMA_IOU_MATCH = 0.20       # slightly higher link threshold for better stability
_prev_boxes: list = []

# ── CLAHE + sharpening ────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_SHARPEN_KERNEL = np.array(
    [[ 0, -0.5,  0],
     [-0.5, 3.0, -0.5],
     [ 0, -0.5,  0]], dtype=np.float32
)

# ── Load ONNX session ─────────────────────────────────────────────────────────
_scrfd_session: ort.InferenceSession | None = None
_INPUT_NAME = "input.1"

if _MODEL_FILE is not None:
    try:
        _scrfd_session = ort.InferenceSession(
            _MODEL_FILE,
            providers=["CPUExecutionProvider"],
        )
        _INPUT_NAME = _scrfd_session.get_inputs()[0].name
        print(
            f"[face_detector] SCRFD-10G loaded ✓  "
            f"(score≥{_SCORE_THRESHOLD}, nms={_NMS_IOU}, "
            f"input={_INPUT_SIZE}px)  [{os.path.basename(_MODEL_FILE)}]"
        )
    except Exception as _e:
        print(f"[face_detector] ERROR loading SCRFD model: {_e}")
else:
    print("[face_detector] WARNING: SCRFD model not found; detection disabled.")


# ── Pre-compute anchor grids ──────────────────────────────────────────────────
def _make_anchor_map(input_size: int) -> dict:
    anchor_map = {}
    for stride in _STRIDES:
        feat = input_size // stride
        gy, gx = np.mgrid[0:feat, 0:feat]
        anchors = np.stack([gx, gy], axis=-1).reshape(-1, 2)
        anchors = np.repeat(anchors, _NUM_ANCHORS, axis=0)
        anchor_map[stride] = anchors.astype(np.float32)
    return anchor_map

_ANCHOR_MAP = _make_anchor_map(_INPUT_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Throttled adaptive CCTV enhancement.
    Runs CLAHE+sharpening every _ENHANCE_EVERY frames; reuses cache between.
    """
    global _enhance_counter, _last_enhanced

    _enhance_counter += 1
    if _last_enhanced is not None and (_enhance_counter % _ENHANCE_EVERY != 0):
        return _last_enhanced

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
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 180:
        enhanced = cv2.filter2D(enhanced, -1, _SHARPEN_KERNEL)

    _last_enhanced = enhanced
    return enhanced


def _preprocess(frame: np.ndarray):
    """Letterbox-resize to _INPUT_SIZE, normalize for SCRFD."""
    h, w  = frame.shape[:2]
    scale = min(_INPUT_SIZE / w, _INPUT_SIZE / h)
    nw    = int(w * scale)
    nh    = int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
    pad[:nh, :nw] = resized

    blob = pad.astype(np.float32)
    blob = (blob - 127.5) / 128.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_iou(boxA, boxB) -> float:
    """IoU between two (x, y, w, h) boxes."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
    return inter / (union + 1e-5)


def _nms(boxes_xywh: list, scores: list, iou_thresh: float) -> list:
    """Greedy NMS. Returns indices of kept boxes (high-score-first order)."""
    if not boxes_xywh:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept  = []
    while order:
        i = order.pop(0)
        kept.append(i)
        order = [j for j in order
                 if _compute_iou(boxes_xywh[i], boxes_xywh[j]) < iou_thresh]
    return kept


def _validate_landmarks(x: int, y: int, w: int, h: int, lm: list) -> bool:
    """
    Require >= 4/5 landmark points inside a ±25%-padded face box.
    Tighter than the old 60% — kills FP landmarks on walls/plants.
    """
    if lm is None or len(lm) < 10:
        return False
    px, py     = w * 0.25, h * 0.25
    x_lo, x_hi = x - px, x + w + px
    y_lo, y_hi = y - py, y + h + py
    valid = sum(
        1 for i in range(5)
        if x_lo <= lm[2 * i] <= x_hi and y_lo <= lm[2 * i + 1] <= y_hi
    )
    return valid >= 4


def _is_partial_face(x: int, y: int, w: int, h: int,
                     frame_w: int, frame_h: int,
                     max_clip_ratio: float = 0.30) -> bool:
    clip_x = max(0, -x) + max(0, (x + w) - frame_w)
    clip_y = max(0, -y) + max(0, (y + h) - frame_h)
    clipped = clip_x * h + clip_y * w
    face_area = max(w * h, 1)
    return (clipped / face_area) > max_clip_ratio


def _face_sharpness(frame: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Return Laplacian variance of the face crop — used as quality gate."""
    fh, fw = frame.shape[:2]
    x1c = max(0, x);     y1c = max(0, y)
    x2c = min(fw, x+w);  y2c = min(fh, y+h)
    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ══════════════════════════════════════════════════════════════════════════════
#  SCRFD INFERENCE + DECODE
# ══════════════════════════════════════════════════════════════════════════════

def _scrfd_infer(frame: np.ndarray,
                 score_thresh: float = _SCORE_THRESHOLD) -> list:
    """
    Run SCRFD on a BGR frame. Returns raw decoded detections in ORIGINAL frame
    coordinates before quality filtering.

    Each element: (x1_f, y1_f, x2_f, y2_f, conf, kps_10floats)
    """
    if _scrfd_session is None:
        return []

    blob, scale = _preprocess(frame)
    outputs = _scrfd_session.run(None, {_INPUT_NAME: blob})

    raw = []
    for i, stride in enumerate(_STRIDES):
        scores  = outputs[i][:, 0]
        bboxes  = outputs[i + 3]
        kps     = outputs[i + 6]
        anchors = _ANCHOR_MAP[stride]

        mask = scores > score_thresh
        if not mask.any():
            continue

        sc  = scores[mask]
        bb  = bboxes[mask]
        kp  = kps[mask]
        anc = anchors[mask]

        cx = (anc[:, 0] + 0.5) * stride
        cy = (anc[:, 1] + 0.5) * stride

        x1 = (cx - bb[:, 0] * stride) / scale
        y1 = (cy - bb[:, 1] * stride) / scale
        x2 = (cx + bb[:, 2] * stride) / scale
        y2 = (cy + bb[:, 3] * stride) / scale

        kps_dec = np.zeros_like(kp)
        for j in range(5):
            kps_dec[:, 2*j]   = (anc[:, 0] + kp[:, 2*j])   * stride / scale
            kps_dec[:, 2*j+1] = (anc[:, 1] + kp[:, 2*j+1]) * stride / scale

        for k in range(len(sc)):
            raw.append((
                float(x1[k]), float(y1[k]),
                float(x2[k]), float(y2[k]),
                float(sc[k]),
                kps_dec[k].tolist(),
            ))

    return raw


def _raw_to_xywh(raw: list, frame_w: int, frame_h: int,
                 min_size: int, frame: np.ndarray | None = None) -> tuple:
    """
    Convert raw detections → (x, y, w, h, lm, conf) tuples after quality gates.
    Returns (faces_list, boxes_xywh_list, scores_list).

    Quality gates applied (in order):
        1. Minimum box size
        2. Partial-face clip ratio
        3. Aspect ratio  0.5 – 1.8  (tighter to kill FP on walls/plants)
        4. Post-decode confidence >= _POST_NMS_THRESH
        5. Landmark validation (>=4/5 within ±25% pad)
        6. Face crop sharpness >= _MIN_SHARPNESS  (if frame provided)
    """
    faces      = []
    boxes_xywh = []
    scores_lst = []

    for (x1, y1, x2, y2, conf, kps) in raw:
        # Gate 4: confidence
        if conf < _POST_NMS_THRESH:
            continue

        x1c = max(0.0, x1);          y1c = max(0.0, y1)
        x2c = min(float(frame_w), x2); y2c = min(float(frame_h), y2)
        bw = x2c - x1c
        bh = y2c - y1c

        # Gate 1: minimum size
        if bw < min_size or bh < min_size:
            continue

        # Gate 2: partial face
        if _is_partial_face(int(x1), int(y1), int(x2 - x1), int(y2 - y1),
                             frame_w, frame_h):
            continue

        # Gate 3: tighter aspect ratio (was 0.4–2.2, now 0.5–1.8)
        aspect = bw / (bh + 1e-5)
        if aspect < 0.45 or aspect > 1.9:
            continue

        ix = int(x1c);  iy = int(y1c)
        iw = int(bw);   ih = int(bh)

        # Gate 5: landmark validation
        lm = kps if _validate_landmarks(ix, iy, iw, ih, kps) else None

        # Gate 6: sharpness (only when frame is supplied)
        if frame is not None and lm is not None:
            sharp = _face_sharpness(frame, ix, iy, iw, ih)
            if sharp < _MIN_SHARPNESS:
                lm = None   # mark as no-landmark but still allow detection box

        faces.append((ix, iy, iw, ih, lm, conf))
        boxes_xywh.append((ix, iy, iw, ih))
        scores_lst.append(conf)

    return faces, boxes_xywh, scores_lst


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_faces_dnn(frame: np.ndarray, smooth: bool = True, min_size: int = 18):
    """
    Detect faces using SCRFD-10G with adaptive CCTV preprocessing.

    Returns: list of (x, y, w, h, landmarks_or_None, conf)
        landmarks = list of 10 floats [rx,ry, lx,ly, nx,ny, rmx,rmy, lmx,lmy]
                    or None if quality gates fail
        x, y, w, h — int, top-left + dimensions
    """
    global _prev_boxes

    if _scrfd_session is None:
        return _prev_boxes if smooth else []

    fh, fw = frame.shape[:2]
    enhanced = _enhance_frame(frame)

    raw = _scrfd_infer(enhanced)

    # Convert + quality filter (pass original frame for sharpness gate)
    current_faces, boxes_xywh, scores_lst = _raw_to_xywh(raw, fw, fh, min_size, enhanced)

    # Strong NMS
    kept = _nms(boxes_xywh, scores_lst, _NMS_IOU)
    current_faces = [current_faces[i] for i in kept]

    if not smooth:
        return current_faces

    # Ghost-box prevention
    if not current_faces:
        _prev_boxes = []
        return []

    # ── EMA tracking ─────────────────────────────────────────────────────────
    smoothed  = []
    unmatched = list(_prev_boxes)

    for c in current_faces:
        c_box  = c[0:4]
        c_rest = c[4:]
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
            smoothed.append(c)

    _prev_boxes = smoothed
    return smoothed


def clear_detector_state():
    """Reset EMA tracking and enhancement cache. Call on camera stop/reset."""
    global _prev_boxes, _last_enhanced, _enhance_counter
    _prev_boxes      = []
    _last_enhanced   = None
    _enhance_counter = 0


def detect_faces_multiscale(frame: np.ndarray, min_size: int = 18) -> list:
    """
    High-recall detection for live streaming and CCTV.

    Pass 1: Standard inference at _INPUT_SIZE.
    Pass 2: 1.5× upscale — ONLY when no/tiny faces found in pass 1 (saves CPU).
    Both passes share global NMS.

    Returns: list of (x, y, w, h, landmarks_or_None, conf)
    """
    if _scrfd_session is None:
        return []

    fh, fw = frame.shape[:2]
    enhanced = _enhance_frame(frame)

    # ── Pass 1 ───────────────────────────────────────────────────────────────
    raw1 = _scrfd_infer(enhanced, score_thresh=_SCORE_THRESHOLD)
    faces1, _, _ = _raw_to_xywh(raw1, fw, fh, min_size, enhanced)

    # ── Pass 2: upscale ONLY when needed ─────────────────────────────────────
    raw2 = []
    if fw > 720 and (len(faces1) == 0 or all(f[2] < 40 for f in faces1)):
        up       = 1.5
        nw2      = min(int(fw * up), 1920)
        nh2      = min(int(fh * up), 1920)
        up_frame = cv2.resize(enhanced, (nw2, nh2), interpolation=cv2.INTER_LINEAR)
        inv      = 1.0 / up
        for (x1, y1, x2, y2, conf, kps) in _scrfd_infer(up_frame,
                                                          score_thresh=_SCORE_THRESHOLD):
            raw2.append((
                x1 * inv, y1 * inv, x2 * inv, y2 * inv,
                conf,
                [v * inv for v in kps],
            ))

    all_raw = raw1 + raw2
    if not all_raw:
        return []

    # ── Global NMS across both passes ────────────────────────────────────────
    faces, boxes_xywh, scores_lst = _raw_to_xywh(all_raw, fw, fh, min_size, enhanced)
    kept = _nms(boxes_xywh, scores_lst, _NMS_IOU)
    return [faces[i] for i in kept]
