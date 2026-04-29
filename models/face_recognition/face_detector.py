"""
face_detector.py  — SCRFD-10G face detector (replaces YuNet)

Model:   det_10g.onnx  (InsightFace SCRFD-10GF with keypoints, already in weights/)
Alias:   scrfd_10g_bnkps.onnx (checked as fallback if renamed copy provided)

Detection pipeline:
    1. Adaptive CLAHE  — only when frame is dark (mean-L < 110) or low-contrast (std-L < 35)
    2. Gentle sharpening — only when Laplacian variance < 180
    3. SCRFD inference at score_threshold=0.40
         3-stride anchor decode (strides 8 / 16 / 32, 2 anchors per cell)
         Landmark decode (5 points → 10 floats, ArcFace order)
    4. Partial-face filter — rejects boxes where > 30% area falls outside frame edge
    5. Aspect-ratio filter — 0.4 – 2.2 (ignores furniture, signage, CCTV artefacts)
    6. Landmark sanity check — requires ≥ 4/5 points inside ±60%-padded face box
    7. Greedy IoU NMS at 0.35 — removes cross-stride duplicates
    8. EMA box smoothing α=0.55 — tracks moving faces without jitter

Return format (identical to old YuNet wrapper — NO other file changes needed):
    list of (x, y, w, h, landmarks_or_None, conf)
        x, y, w, h   — int, top-left origin + dimensions
        landmarks    — list of 10 floats [rx,ry, lx,ly, nx,ny, rmx,rmy, lmx,lmy]
                       or None if landmark validation fails
        conf         — float [0, 1]

SCRFD landmark order is identical to ArcFace reference landmarks:
    index 0,1  → right eye       _ARCFACE_DST[0]  [38.29, 51.70]
    index 2,3  → left  eye       _ARCFACE_DST[1]  [73.53, 51.50]
    index 4,5  → nose tip        _ARCFACE_DST[2]  [56.03, 71.74]
    index 6,7  → right mouth     _ARCFACE_DST[3]  [41.55, 92.37]
    index 8,9  → left  mouth     _ARCFACE_DST[4]  [70.73, 92.20]
No reordering required.
"""

import cv2
import numpy as np
import os
import onnxruntime as ort

# ── Model paths (primary = InsightFace default name; alias = user-requested name) ─
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_CANDIDATES = [
    os.path.join(_BASE_DIR, "weights", "det_10g.onnx"),
    os.path.join(_BASE_DIR, "weights", "scrfd_10g_bnkps.onnx"),
]

_MODEL_FILE = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)), None)

# ── Detection hyper-parameters ────────────────────────────────────────────────
_SCORE_THRESHOLD = 0.40   # recall-precision balance: 0.40 catches side/far/CCTV faces
_NMS_IOU         = 0.35   # tight NMS — kills cross-stride duplicates reliably
_INPUT_SIZE      = 640    # SCRFD-10G standard; good balance of speed vs far-face recall
_STRIDES         = [8, 16, 32]
_NUM_ANCHORS     = 2      # SCRFD-10G uses 2 anchors per grid cell

# ── EMA smoothing (moving-face tracking) ──────────────────────────────────────
_BOX_ALPHA     = 0.55   # weight on previous position (lower = more responsive)
_EMA_IOU_MATCH = 0.15   # min IoU to link new detection → existing track
_prev_boxes: list = []

# ── CLAHE + sharpening (CCTV pre-processing, identical to old wrapper) ─────────
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_SHARPEN_KERNEL = np.array(
    [[ 0, -0.5,  0],
     [-0.5, 3.0, -0.5],
     [ 0, -0.5,  0]], dtype=np.float32
)

# ── Load ONNX session ─────────────────────────────────────────────────────────
_scrfd_session: ort.InferenceSession | None = None
_INPUT_NAME = "input.1"   # confirmed from model inspection

if _MODEL_FILE is not None:
    try:
        _scrfd_session = ort.InferenceSession(
            _MODEL_FILE,
            providers=["CPUExecutionProvider"],
        )
        _INPUT_NAME = _scrfd_session.get_inputs()[0].name
        print(
            f"[face_detector] Loaded SCRFD-10G ✓  "
            f"(score≥{_SCORE_THRESHOLD}, nms_iou={_NMS_IOU}, "
            f"input={_INPUT_SIZE}px)  [{os.path.basename(_MODEL_FILE)}]"
        )
    except Exception as _e:
        print(f"[face_detector] ERROR loading SCRFD model: {_e}")
else:
    print("[face_detector] WARNING: SCRFD model not found (det_10g.onnx / scrfd_10g_bnkps.onnx); detection disabled.")


# ── Pre-compute anchor grids (done once at import) ────────────────────────────
def _make_anchor_map(input_size: int) -> dict:
    """Return {stride: ndarray(N,2)} of (gx, gy) anchor grid coordinates."""
    anchor_map = {}
    for stride in _STRIDES:
        feat = input_size // stride          # 80, 40, 20
        gy, gx = np.mgrid[0:feat, 0:feat]
        anchors = np.stack([gx, gy], axis=-1).reshape(-1, 2)   # (feat*feat, 2)
        anchors = np.repeat(anchors, _NUM_ANCHORS, axis=0)      # 2 anchors / cell
        anchor_map[stride] = anchors.astype(np.float32)
    return anchor_map

_ANCHOR_MAP = _make_anchor_map(_INPUT_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Adaptive CCTV enhancement — only touches frames that actually need it.
    CLAHE when: mean-L < 110 (dark) OR std-L < 35 (low contrast)
    Sharpening when: Laplacian variance < 180 (blurry)
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
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 180:
        enhanced = cv2.filter2D(enhanced, -1, _SHARPEN_KERNEL)

    return enhanced


def _preprocess(frame: np.ndarray):
    """
    Letterbox-resize to _INPUT_SIZE × _INPUT_SIZE, normalize for SCRFD.
    Returns: (blob NCHW float32, scale float)
    """
    h, w  = frame.shape[:2]
    scale = min(_INPUT_SIZE / w, _INPUT_SIZE / h)
    nw    = int(w * scale)
    nh    = int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
    pad[:nh, :nw] = resized

    blob = pad.astype(np.float32)
    blob = (blob - 127.5) / 128.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]   # → NCHW
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
    Require ≥ 4/5 landmark points inside a ±60%-padded face box.
    Rejects non-face objects whose keypoints scatter outside the box.
    """
    if lm is None or len(lm) < 10:
        return False
    px, py     = w * 0.60, h * 0.60
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
    """
    Return True when > max_clip_ratio of the face box is clipped by the frame boundary.
    Partial/occluded faces at the frame edge produce poor ArcFace embeddings.
    """
    clip_x = max(0, -x) + max(0, (x + w) - frame_w)
    clip_y = max(0, -y) + max(0, (y + h) - frame_h)
    # Conservative upper bound on clipped area
    clipped = clip_x * h + clip_y * w
    face_area = max(w * h, 1)
    return (clipped / face_area) > max_clip_ratio


# ══════════════════════════════════════════════════════════════════════════════
#  SCRFD INFERENCE + DECODE
# ══════════════════════════════════════════════════════════════════════════════

def _scrfd_infer(frame: np.ndarray,
                 score_thresh: float = _SCORE_THRESHOLD) -> list:
    """
    Run SCRFD on a BGR frame. Returns raw decoded detections in ORIGINAL frame
    coordinates before any quality filtering.

    Each element: (x1_f, y1_f, x2_f, y2_f, conf, kps_10floats)
        x1/y1/x2/y2  — float, original-frame pixel coords (may be slightly outside bounds)
        conf         — float [0, 1]
        kps_10floats — [rx,ry, lx,ly, nx,ny, rmx,rmy, lmx,lmy]  (SCRFD = ArcFace order)
    """
    if _scrfd_session is None:
        return []

    blob, scale = _preprocess(frame)
    outputs = _scrfd_session.run(None, {_INPUT_NAME: blob})

    # Layout: outputs[0:3] scores (N,1) | outputs[3:6] bboxes (N,4) | outputs[6:9] kps (N,10)
    raw = []
    for i, stride in enumerate(_STRIDES):
        scores  = outputs[i][:, 0]       # (N,)
        bboxes  = outputs[i + 3]         # (N, 4)
        kps     = outputs[i + 6]         # (N, 10)
        anchors = _ANCHOR_MAP[stride]    # (N, 2) — (gx, gy)

        mask = scores > score_thresh
        if not mask.any():
            continue

        sc  = scores[mask]
        bb  = bboxes[mask]
        kp  = kps[mask]
        anc = anchors[mask]

        # Anchor centres in padded-input space
        cx = (anc[:, 0] + 0.5) * stride
        cy = (anc[:, 1] + 0.5) * stride

        # Decode bounding box (distance offsets → absolute padded coords → original)
        x1 = (cx - bb[:, 0] * stride) / scale
        y1 = (cy - bb[:, 1] * stride) / scale
        x2 = (cx + bb[:, 2] * stride) / scale
        y2 = (cy + bb[:, 3] * stride) / scale

        # Decode keypoints (grid offsets → absolute padded coords → original)
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
                 min_size: int) -> tuple:
    """
    Convert raw detections → (x, y, w, h, lm, conf) tuples after quality gates.
    Returns (faces_list, boxes_xywh_list, scores_list) ready for NMS.
    """
    faces      = []
    boxes_xywh = []
    scores_lst = []

    for (x1, y1, x2, y2, conf, kps) in raw:
        # Clamp to frame
        x1c = max(0.0, x1);        y1c = max(0.0, y1)
        x2c = min(float(frame_w), x2); y2c = min(float(frame_h), y2)

        bw = x2c - x1c
        bh = y2c - y1c

        # Minimum size gate
        if bw < min_size or bh < min_size:
            continue

        # Reject heavily-clipped partial faces at frame borders
        if _is_partial_face(int(x1), int(y1), int(x2 - x1), int(y2 - y1),
                             frame_w, frame_h):
            continue

        # Aspect ratio — human face: roughly square to portrait
        aspect = bw / (bh + 1e-5)
        if aspect < 0.4 or aspect > 2.2:
            continue

        ix = int(x1c);  iy = int(y1c)
        iw = int(bw);   ih = int(bh)

        lm = kps if _validate_landmarks(ix, iy, iw, ih, kps) else None

        faces.append((ix, iy, iw, ih, lm, conf))
        boxes_xywh.append((ix, iy, iw, ih))
        scores_lst.append(conf)

    return faces, boxes_xywh, scores_lst


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  — identical signatures to the old YuNet wrapper
# ══════════════════════════════════════════════════════════════════════════════

def get_faces_dnn(frame: np.ndarray, smooth: bool = True, min_size: int = 20):
    """
    Detect faces using SCRFD-10G with adaptive CCTV preprocessing.

    Returns: list of (x, y, w, h, landmarks_or_None, conf)
        landmarks = list of 10 floats [rx,ry, lx,ly, nx,ny, rmx,rmy, lmx,lmy]
                    or None if landmark validation fails
        x, y, w, h — int, (top-left, width, height)  ← same as old YuNet output
    """
    global _prev_boxes

    if _scrfd_session is None:
        return _prev_boxes if smooth else []

    fh, fw = frame.shape[:2]
    enhanced = _enhance_frame(frame)

    raw = _scrfd_infer(enhanced)

    # Convert + quality filter
    current_faces, boxes_xywh, scores_lst = _raw_to_xywh(raw, fw, fh, min_size)

    # NMS
    kept = _nms(boxes_xywh, scores_lst, _NMS_IOU)
    current_faces = [current_faces[i] for i in kept]

    if not smooth:
        return current_faces

    # Ghost-box prevention
    if not current_faces:
        _prev_boxes = []
        return []

    # ── EMA tracking — link new detections to previous positions ─────────────
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
            smoothed.append(c)   # new face — no prior to blend

    _prev_boxes = smoothed
    return smoothed


def clear_detector_state():
    """Reset EMA tracking state. Call on camera stop or model reset."""
    global _prev_boxes
    _prev_boxes = []


def detect_faces_multiscale(frame: np.ndarray, min_size: int = 40) -> list:
    """
    High-recall detection for live streaming and CCTV.

    SCRFD-10G natively handles multiple scales via its 3-stride architecture
    (stride 8 = near/large faces, stride 32 = far/small faces), so a single
    640-px inference already replaces the old YuNet multi-scale loop.

    An additional 1.5× upscale pass is performed for frames > 480 px to
    improve recall on very distant faces (e.g. far-end CCTV).
    Both passes share global NMS to eliminate cross-pass duplicates.

    Returns: list of (x, y, w, h, landmarks_or_None, conf)
    """
    if _scrfd_session is None:
        return []

    fh, fw = frame.shape[:2]
    enhanced = _enhance_frame(frame)

    # ── Pass 1: standard 640-px inference ────────────────────────────────────
    raw1 = _scrfd_infer(enhanced, score_thresh=_SCORE_THRESHOLD)

    # ── Pass 2: 1.5× upscale for far/small faces (CCTV wide-angle shots) ────
    raw2 = []
    if max(fw, fh) > 480:
        up   = 1.5
        nw2  = min(int(fw * up), 1920)
        nh2  = min(int(fh * up), 1920)
        up_frame = cv2.resize(enhanced, (nw2, nh2), interpolation=cv2.INTER_LINEAR)
        inv  = 1.0 / up
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

    # ── Convert + quality filter + global NMS ────────────────────────────────
    faces, boxes_xywh, scores_lst = _raw_to_xywh(all_raw, fw, fh, min_size)
    kept = _nms(boxes_xywh, scores_lst, _NMS_IOU)
    return [faces[i] for i in kept]
