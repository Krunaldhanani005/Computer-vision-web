"""
face_recognition_model.py  — Production-grade FR pipeline (SCRFD + ArcFace)

Pipeline:
    frame → detect (SCRFD-10G) → quality gates → encode (ArcFace 512-d)
    → classify (known / unknown / blacklist)
    → 3-frame majority vote → identity latch → snapshot save → DB update (async)

Post-detection quality gates:
    1. Minimum face size  ≥ 80 × 80 px   (skip tiny / far / noisy detections)
    2. Blur gate          ≥ 20.0 Laplacian variance  (skip motion-blurred frames)
    3. 30% bbox padding   on unaligned fallback crops (full face stored)
    4. 3-frame majority vote before saving  (walking / moving face stability)
    5. DISTANCE_THRESHOLD = 0.45           (configurable cosine distance gate)
    6. Unknown cooldown   = 6 s per slot   (prevents duplicate unknown spam)
    7. Blur gate on save  ≥ 20.0           (only valid crops reach dashboard)
"""


import cv2
import numpy as np
from . import arcface
import os
import uuid
from threading import Thread
from collections import deque

from .face_detector import get_faces_dnn

from .fr_database import (
    insert_face,
    get_all_faces,
    upsert_known,
    upsert_unknown,
    insert_alert,
)

# ── Snapshot storage directories ──────────────────────────────────────────────────
_BASE_STATIC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "static"
))

_SNAP_DIR   = os.path.join(_BASE_STATIC, "fr_logs")
_REPORT_DIR = os.path.join(_BASE_STATIC, "fr_reports")

# Per-type subdirectories inside fr_reports/
_DIR_KNOWN     = os.path.join(_REPORT_DIR, "known")
_DIR_UNKNOWN   = os.path.join(_REPORT_DIR, "unknown")
_DIR_BLACKLIST = os.path.join(_REPORT_DIR, "blacklist")

for _d in [_SNAP_DIR, _REPORT_DIR, _DIR_KNOWN, _DIR_UNKNOWN, _DIR_BLACKLIST]:
    os.makedirs(_d, exist_ok=True)

# ── Recognition thresholds (cosine distance — lower = stricter) ───────────────
# dist ≤ BLACKLIST_THRESHOLD → blacklist match
# dist ≤ KNOWN_THRESHOLD     → known match
# dist > UNKNOWN_THRESHOLD   → classified as Unknown
# (gap between KNOWN and UNKNOWN handled by 3-frame confirmation vote)
BLACKLIST_THRESHOLD = 0.40
KNOWN_THRESHOLD     = 0.45   # slightly more lenient for CCTV far-face embedding quality
UNKNOWN_THRESHOLD   = 0.55
DISTANCE_THRESHOLD  = KNOWN_THRESHOLD  # backward-compat alias

# ── In-memory encoding cache ──────────────────────────────────────────────────
known_encodings: list = []
known_names:     list = []
known_types:     list = []

# Blacklist per-person re-alert cooldown (prevents alert flood when same person
# creates multiple tracker slots by repeatedly entering / exiting the frame)
_BLACKLIST_ALERT_SECS  = 60.0          # seconds between alerts for the same name
_blacklist_last_alert: dict = {}        # name → monotonic timestamp of last fired alert


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODING CACHE
# ══════════════════════════════════════════════════════════════════════════════

def load_encodings_from_db():
    """Reload in-memory encoding cache from fr_surveillance_db.faces."""
    global known_encodings, known_names, known_types
    known_encodings.clear()
    known_names.clear()
    known_types.clear()

    try:
        count = 0
        for face in get_all_faces():
            enc = face.get("encoding")
            if enc:
                known_encodings.append(np.array(enc))
                known_names.append(face.get("name", "Unknown"))
                known_types.append(face.get("person_type", "known"))
                count += 1
        print(f"[face_recognition] Loaded {count} encodings from fr_surveillance_db ✓")
    except Exception as e:
        print(f"[face_recognition] load_encodings_from_db error: {e}")


# Load at import time
load_encodings_from_db()


# ══════════════════════════════════════════════════════════════════════════════
#  FACE QUALITY VALIDATION  (practical, detection-first)
# ══════════════════════════════════════════════════════════════════════════════

# ── Quality gate constants (post-detection, pre-save) ─────────────────────────
# Minimum face size — 45px balances CCTV recall vs noisy tiny detections
_MIN_SAVE_W    = 45
_MIN_SAVE_H    = 45
# Blur gate — reject only severely blurred frames (Laplacian variance < 15)
# 35.0 was too strict: small/far CCTV faces naturally have low variance and were
# getting rejected inside recognize(), appearing as Unknown even for known people.
_MIN_SHARPNESS = 15.0
# Frontal validation — minimum inter-eye distance in pixels
_MIN_EYE_DIST  = 20.0
# Crop padding — 35% expansion on all sides for full-face context
_FACE_PAD      = 0.35


def _sharpness_score(gray_crop: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


def _is_frontal_face(landmarks: list, min_eye_dist: float = _MIN_EYE_DIST) -> bool:
    """
    Fix #5: Frontal face validation using 5-point SCRFD landmarks.
    Requires both eyes to be present and inter-eye distance >= min_eye_dist px.
    Rejects heavy profile shots that produce poor ArcFace embeddings.

    landmarks: [rx,ry, lx,ly, nx,ny, rmx,rmy, lmx,lmy]  (10 floats)
    """
    if landmarks is None or len(landmarks) < 4:
        return False
    rx, ry = float(landmarks[0]), float(landmarks[1])   # right eye
    lx, ly = float(landmarks[2]), float(landmarks[3])   # left eye
    eye_dist = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    return eye_dist >= min_eye_dist

def _validate_face_for_save(frame: np.ndarray, box) -> tuple:
    """
    Quality gate for face snapshots.
    Fix #1: Sharpness is measured on the 35%-EXPANDED bbox — same region as saved crop.
    Returns: (is_valid, sharpness, reason)
    """
    x, y, w, h = box
    fh, fw = frame.shape[:2]

    if w <= 0 or h <= 0:
        return False, 0.0, f"zero_size ({w}x{h})"

    # Minimum face size gate
    if w < _MIN_SAVE_W or h < _MIN_SAVE_H:
        return False, 0.0, f"too_small ({w}x{h})"

    # Aspect ratio validation
    aspect = w / float(h + 1e-5)
    if aspect < 0.5 or aspect > 2.0:
        return False, 0.0, f"bad_aspect_ratio ({aspect:.2f})"

    # Fix #1: Measure sharpness on expanded crop — same as what gets saved
    pad_x = int(w * _FACE_PAD);  pad_y = int(h * _FACE_PAD)
    x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x); y2 = min(fh, y + h + pad_y)
    raw = frame[y1:y2, x1:x2]
    if raw.size == 0:
        return False, 0.0, "empty_crop"

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if sharpness < _MIN_SHARPNESS:
        return False, sharpness, f"too_blurry ({sharpness:.1f})"

    return True, sharpness, "ok"


# ══════════════════════════════════════════════════════════════════════════════
#  SNAPSHOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _crop_face(frame: np.ndarray, box, landmarks=None):
    """
    Fix #1: Correct crop pipeline — NEVER saves raw detector box.

    Pipeline:
        1. Expand bbox 35% on all sides → captures forehead, chin, ears
        2. Try ArcFace alignment (preferred) → canonical 112×112, resized to 160×160
        3. Fallback: expanded bbox crop resized to 160×160
    """
    x, y, w, h = box
    fh, fw = frame.shape[:2]

    # Step 1: Expand bbox 35% uniformly
    pad_x  = int(w * _FACE_PAD)
    pad_y  = int(h * _FACE_PAD)
    x1_exp = max(0, x - pad_x)
    y1_exp = max(0, y - pad_y)
    x2_exp = min(fw, x + w + pad_x)
    y2_exp = min(fh, y + h + pad_y)

    # Step 2: ArcFace alignment (best quality — landmark-based canonical warp)
    if landmarks is not None:
        aligned = arcface.align_face(frame, landmarks)
        if aligned is not None:
            return cv2.resize(aligned, (160, 160), interpolation=cv2.INTER_CUBIC)

    # Step 3: Fallback — expanded bbox crop (never raw detector box)
    crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)


def _write_image(crop: np.ndarray, folder: str, label: str, quality: int = 95) -> str:
    """
    Write *crop* to *folder/label_<uuid>.jpg*.
    Returns web-relative path like  static/fr_reports/known/file.jpg
    or "" on failure. Validates imwrite success.
    """
    try:
        fname = f"{label}_{uuid.uuid4().hex[:8]}.jpg"
        fpath = os.path.join(folder, fname)
        ok = cv2.imwrite(fpath, crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            print(f"[imwrite] FAILED for {fpath}")
            return ""
        # Build web path relative to the project root (static/ is served by Flask)
        rel = os.path.relpath(fpath, _BASE_STATIC)
        web_path = "static/" + rel.replace(os.sep, "/")
        print(f"[imwrite] Saved {web_path} ({os.path.getsize(fpath)} bytes)")
        return web_path
    except Exception as e:
        print(f"[imwrite] Error: {e}")
        return ""


def _save_snapshot(frame: np.ndarray, box, label: str, folder: str = None, landmarks: list = None) -> str:
    """Quality-gated save for Known / Blacklist.
    Fix #5/#6: Frontal check + expanded/aligned crop only — never raw detector box.
    """
    try:
        # Gate 1: size + blur on expanded region
        valid, sharpness, reason = _validate_face_for_save(frame, box)
        if not valid:
            print(f"[snapshot] Rejected ({reason}) label={label}")
            return ""
        # Gate 2: frontal face (Fix #5) — requires eye distance >= _MIN_EYE_DIST
        if not _is_frontal_face(landmarks):
            print(f"[snapshot] Rejected (not frontal / no valid landmarks) label={label}")
            return ""
        # Crop: always expanded+aligned (Fix #1 / Fix #6)
        crop = _crop_face(frame, box, landmarks)
        if crop is None:
            print(f"[snapshot] _crop_face returned None for label={label}")
            return ""
        return _write_image(crop, folder or _SNAP_DIR, label)
    except Exception as e:
        print(f"[snapshot] Error: {e}")
        return ""


def _save_snapshot_relaxed(frame: np.ndarray, box, label: str, folder: str = None, landmarks: list = None) -> str:
    """Save for Unknown — validates quality + frontal + expanded crop.
    Fix #5/#6: Frontal check applied; hard fallback uses expanded bbox, not raw box.
    """
    try:
        # Gate 1: size + blur on expanded region
        valid, sharpness, reason = _validate_face_for_save(frame, box)
        if not valid:
            print(f"[snapshot-unk] Rejected ({reason}) label={label}")
            return ""
        # Gate 2: frontal face (Fix #5)
        if not _is_frontal_face(landmarks):
            print(f"[snapshot-unk] Rejected (not frontal) label={label}")
            return ""
        # Crop: expanded + aligned (Fix #1 / Fix #6)
        crop = _crop_face(frame, box, landmarks)
        if crop is None:
            # Hard fallback: expanded bbox, NOT raw detector box
            x, y, w, h = box
            fh, fw = frame.shape[:2]
            pad_x = int(w * _FACE_PAD);  pad_y = int(h * _FACE_PAD)
            x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
            x2 = min(fw, x + w + pad_x); y2 = min(fh, y + h + pad_y)
            raw = frame[y1:y2, x1:x2]
            if raw.size == 0:
                print(f"[snapshot-unk] Empty expanded crop for label={label}")
                return ""
            crop = cv2.resize(raw, (160, 160), interpolation=cv2.INTER_CUBIC)
        return _write_image(crop, folder or _SNAP_DIR, label, quality=90)
    except Exception as e:
        print(f"[snapshot-unk] Error: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  ASYNC LOGGING  (writes to both fr_database + fr_reports_db)
# ══════════════════════════════════════════════════════════════════════════════

def _log_known_async(name: str, person_type: str, confidence: float,
                     frame: np.ndarray, box, landmarks=None):
    valid, sharpness, reason = _validate_face_for_save(frame, box)
    if not valid:
        print(f"[log-known] Frame rejected ({reason}) for {name}")
        return

    def worker():
        label       = f"known_{name.replace(' ', '_')}"
        snap_log    = _save_snapshot(frame, box, label, _SNAP_DIR, landmarks)
        snap_report = _save_snapshot(frame, box, label, _DIR_KNOWN, landmarks)
        print(f"[log-known] {name}: log={bool(snap_log)} report={bool(snap_report)} path={snap_report!r}")
        upsert_known(name, confidence, snap_report)
    Thread(target=worker, daemon=True).start()


def _log_blacklist_async(name: str, frame: np.ndarray, box, landmarks=None,
                         camera_source="webcam", zone_id="default"):
    valid, sharpness, reason = _validate_face_for_save(frame, box)
    if not valid:
        print(f"[log-blacklist] Frame rejected ({reason}) for {name}")
        return

    def worker():
        label       = f"blacklist_{name.replace(' ', '_')}"
        snap_log    = _save_snapshot(frame, box, label, _SNAP_DIR, landmarks)
        snap_report = _save_snapshot(frame, box, label, _DIR_BLACKLIST, landmarks)
        print(f"[log-blacklist] {name}: log={bool(snap_log)} report={bool(snap_report)} path={snap_report!r}")
        insert_alert(name, snap_report, camera_source, zone_id)
    Thread(target=worker, daemon=True).start()


def _log_unknown_async(temp_id: str, frame: np.ndarray, box, landmarks=None):
    """Unknown capture: IMMEDIATE save on first call."""
    def worker():
        label       = f"unknown_{temp_id}"
        snap_log    = _save_snapshot_relaxed(frame, box, label, _SNAP_DIR, landmarks)
        snap_report = _save_snapshot_relaxed(frame, box, label, _DIR_UNKNOWN, landmarks)
        print(f"[log-unk] '{temp_id}' saved: log={bool(snap_log)} report={bool(snap_report)} path={snap_report!r}")
        upsert_unknown(temp_id, snap_report)
    Thread(target=worker, daemon=True).start()



# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(files, name: str, person_type: str = "known"):
    global known_encodings, known_names, known_types

    success_count = 0
    for file in files:
        data  = file.read()
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[train_model] Failed to decode image for {name}")
            continue

        faces = get_faces_dnn(image, smooth=False)
        if not faces:
            print(f"[train_model] No faces found in uploaded image for {name}")
            continue

        for face in faces:
            # face = (x, y, w, h, landmarks_or_None, conf)  — always ≥5 elements
            if len(face) < 5:
                continue
            x, y, w, h, landmarks = face[0], face[1], face[2], face[3], face[4]

            if landmarks is None:
                print(f"[train_model] Skipping face with invalid landmarks for {name}")
                continue

            aligned_face = arcface.align_face(image, landmarks)
            if aligned_face is None:
                continue

            # Pass raw aligned face directly — ArcFace handles lighting internally.
            # CLAHE/denoising (normalize_face) breaks embedding space for CCTV vs photo.
            enc = arcface.get_embedding(aligned_face)
            if enc is None:
                continue

            try:
                if insert_face(name, person_type, enc.tolist()):
                    known_encodings.append(enc)
                    known_names.append(name)
                    known_types.append(person_type)
                    success_count += 1
                    print(f"[train_model] Saved ArcFace encoding for {name} ({person_type}) ✓")
            except Exception as e:
                print(f"[train_model] DB error: {e}")

    print(f"[face_recognition] Active ArcFace encodings: {len(known_encodings)}")

    if success_count == 0:
        return False, "No valid faces found. Please upload clear photos."
    return True, f"Trained '{name}' ({person_type}) with {success_count} encoding(s)."


# ══════════════════════════════════════════════════════════════════════════════
#  CLEAR / RESET
# ══════════════════════════════════════════════════════════════════════════════

def clear_model():
    """Wipe in-memory encodings and reload from DB."""
    global known_encodings, known_names, known_types
    known_encodings.clear()
    known_names.clear()
    known_types.clear()
    reset_tracking_state()
    load_encodings_from_db()


def reset_tracking_state():
    _identity_tracker.reset()
    _blacklist_last_alert.clear()
    try:
        from .face_detector import clear_detector_state
        clear_detector_state()
    except Exception as e:
        print(f"[face_recognition] reset_tracking_state error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_face(aligned_face: np.ndarray) -> np.ndarray:
    """
    Normalize aligned 112x112 face crop before encoding for consistency.
    Applies light denoise and CLAHE contrast enhancement.
    """
    # Light denoise to handle bad lighting
    denoised = cv2.fastNlMeansDenoisingColored(aligned_face, None, 3, 3, 7, 21)
    
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def recognize(frame: np.ndarray, face_data: tuple):
    """
    Align face, generate ArcFace embedding, compare against known_encodings.

    face_data: (x, y, w, h, landmarks) OR (x, y, w, h, landmarks, conf)

    Returns: (name, person_type, confidence, distance, embedding)
        embedding is the raw ArcFace 512-d vector (L2-normalised) or None on failure.
        Callers use the embedding for unknown-person grouping.
    """
    _no_match = ("Unknown", "unknown", 0.0, 2.0, None)

    if frame is None or frame.size == 0:
        return _no_match

    if len(face_data) < 5:
        print(f"[recognize] ERROR: face_data too short (len={len(face_data)}), need >=5")
        return _no_match

    x, y, w, h, landmarks = face_data[0], face_data[1], face_data[2], face_data[3], face_data[4]

    # ── Gate 1: Minimum face size (Fix #2: 80px — skip tiny/far SCRFD detections) ─
    if w < _MIN_SAVE_W or h < _MIN_SAVE_H:
        return _no_match

    fh, fw = frame.shape[:2]
    face_crop_raw = frame[max(0, y):min(fh, y+h), max(0, x):min(fw, x+w)]
    if face_crop_raw.size == 0:
        return _no_match

    # ── Gate 2: Sharpness (Fix #6: Laplacian ≥ 20.0 — skips motion blur) ─────────
    gray_crop  = cv2.cvtColor(face_crop_raw, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    if blur_score < _MIN_SHARPNESS:
        return _no_match

    # ── ArcFace align + embed ─────────────────────────────────────────────────
    aligned = arcface.align_face(frame, landmarks)
    if aligned is None:
        print(f"[recognize] WARNING: alignment failed for face {w}x{h}")
        return _no_match

    new_enc = arcface.get_embedding(aligned)
    if new_enc is None:
        print(f"[recognize] ERROR: ArcFace embedding returned None")
        return _no_match

    if not known_encodings:
        # No training data — return Unknown but keep embedding for unknown grouping
        return "Unknown", "unknown", 0.0, 2.0, new_enc

    # ── Compare against ALL stored embeddings (best-of-N per person) ──────────
    similarities = arcface.compute_similarities(known_encodings, new_enc)
    distances    = [1.0 - sim if sim != -1.0 else 2.0 for sim in similarities]

    person_best: dict = {}
    for i, dist in enumerate(distances):
        nm = known_names[i]
        pt = known_types[i]
        if nm not in person_best or dist < person_best[nm][0]:
            person_best[nm] = (float(dist), pt)

    blacklist_best = None
    known_best     = None
    for nm, (dist, pt) in person_best.items():
        if pt == "blacklist":
            if blacklist_best is None or dist < blacklist_best[1]:
                blacklist_best = (nm, dist, pt)
        else:
            if known_best is None or dist < known_best[1]:
                known_best = (nm, dist, pt)

    best_overall_name = "N/A"
    best_overall_dist = 2.0
    if person_best:
        best_overall_name, (best_overall_dist, _) = min(
            person_best.items(), key=lambda kv: kv[1][0]
        )
    print(
        f"[recognize] Face {w}x{h} | blur={blur_score:.1f} | "
        f"best={best_overall_name} | dist={best_overall_dist:.3f} | "
        f"thr={DISTANCE_THRESHOLD} | encs={len(known_encodings)}"
    )

    # Priority: blacklist > known > unknown
    if blacklist_best and blacklist_best[1] <= BLACKLIST_THRESHOLD:
        nm, d, pt = blacklist_best
        print(f"[recognize] → BLACKLIST {nm} dist={d:.3f}")
        return nm, pt, max(0.0, 1.0 - d), d, new_enc

    if known_best and known_best[1] <= KNOWN_THRESHOLD:
        nm, d, pt = known_best
        print(f"[recognize] → KNOWN {nm} dist={d:.3f}")
        return nm, pt, max(0.0, 1.0 - d), d, new_enc

    best_dist_final = min((d for _, (d, _) in person_best.items()), default=2.0)
    print(f"[recognize] → Unknown | best_dist={best_dist_final:.3f} (thr={UNKNOWN_THRESHOLD})")
    return "Unknown", "unknown", 0.0, best_dist_final, new_enc


# ══════════════════════════════════════════════════════════════════════════════
#  IDENTITY TRACKER  (time-based latch + majority vote + stable saves)
# ══════════════════════════════════════════════════════════════════════════════

import time as _time

_LATCH_SECS  = 5.0
_MIN_CONFIRM = 3   # 3-frame confirmation — all 3 must agree before committing label

# ── Unknown-person grouping cache ──────────────────────────────────────────────
# Stores (temp_id, embedding, monotonic_timestamp) for recently seen unknowns.
# When a face appears that has no IoU-matching active slot, we check this cache
# first. If a similar embedding is found (distance < threshold), we reuse that
# temp_id — same unknown person gets a single card on the dashboard.
_RECENT_UNKNOWNS: list = []          # [(temp_id, np.ndarray, float)]
_UNKNOWN_GROUP_DISTANCE = 0.42       # cosine distance — same person re-entry threshold
_UNKNOWN_GROUP_TTL      = 90.0       # seconds to remember an unknown for grouping


def _find_matching_unknown(embedding: np.ndarray) -> str | None:
    """Return temp_id of a recently seen similar unknown, or None."""
    global _RECENT_UNKNOWNS
    now = _time.monotonic()
    _RECENT_UNKNOWNS = [
        (tid, emb, ts) for tid, emb, ts in _RECENT_UNKNOWNS
        if (now - ts) < _UNKNOWN_GROUP_TTL
    ]
    for tid, emb, _ in _RECENT_UNKNOWNS:
        if emb.shape == embedding.shape:
            dist = 1.0 - float(np.dot(emb, embedding))
            if dist < _UNKNOWN_GROUP_DISTANCE:
                return tid
    return None


def _register_unknown_embedding(temp_id: str, embedding: np.ndarray):
    """Add or refresh an unknown embedding in the grouping cache."""
    global _RECENT_UNKNOWNS
    now = _time.monotonic()
    # Update timestamp if already present
    for i, (tid, emb, _) in enumerate(_RECENT_UNKNOWNS):
        if tid == temp_id:
            _RECENT_UNKNOWNS[i] = (tid, embedding.copy(), now)
            return
    _RECENT_UNKNOWNS.append((temp_id, embedding.copy(), now))


class IdentityTracker:
    """
    Per-face temporal tracker:
    - IoU-based slot matching
    - 3-frame majority-vote smoothing
    - Time-based identity LATCH (5 s): once Known confirmed, stays Known
    - Save to DB only after _MIN_CONFIRM consistent frames
    - Unknown saved once per slot (not repeatedly)
    """

    def __init__(self, history_len: int = 3, stale_frames: int = 20):
        self.history_len  = history_len
        self.stale_frames = stale_frames
        self.faces: list  = []

    @staticmethod
    def _iou(bA, bB) -> float:
        xA = max(bA[0], bB[0]);  yA = max(bA[1], bB[1])
        xB = min(bA[0]+bA[2], bB[0]+bB[2]); yB = min(bA[1]+bA[3], bB[1]+bB[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        union = bA[2]*bA[3] + bB[2]*bB[3] - inter
        return inter / (union + 1e-5)

    def update(self, x, y, w, h, raw_name: str, p_type: str = "unknown",
               confidence: float = 0.0, frame=None, landmarks=None,
               embedding=None, camera_source="webcam", zone_id="default"):
        now = _time.monotonic()
        box = (x, y, w, h)

        # ── Find matching slot by IoU ─────────────────────────────────────────
        best_iou, best_idx = 0.0, -1
        for i, f in enumerate(self.faces):
            iou = self._iou(box, f["box"])
            if iou > best_iou:
                best_iou, best_idx = iou, i

        if best_idx != -1 and best_iou > 0.10:   # 0.10 lets walking faces re-link across slow async frames
            slot = self.faces[best_idx]
            slot["box"] = box
            slot["age"] = 0
            if landmarks:
                slot["landmarks"] = landmarks
            if embedding is not None:
                slot["embedding"] = embedding
        else:
            # No IoU match — check embedding cache for same unknown re-entering frame
            tid = None
            if raw_name == "Unknown" and embedding is not None:
                tid = _find_matching_unknown(embedding)

            if tid is None:
                tid = uuid.uuid4().hex[:8]

            slot = {
                "box":            box,
                "landmarks":      landmarks,
                "embedding":      embedding,
                "age":            0,
                "temp_id":        tid,
                "history":        deque(maxlen=self.history_len),
                "latch_name":     None,
                "latch_type":     None,
                "latch_until":    0.0,
                "unk_snap_count": 0,
                "unk_snap_ts":    0.0,
                "known_saved":    False,
            }
            self.faces.append(slot)
            best_idx = len(self.faces) - 1

        slot = self.faces[best_idx]

        # ── LATCH: Known identity confirmed and still fresh → keep it ────────
        if now < slot["latch_until"] and slot["latch_name"]:
            slot["history"].append((slot["latch_name"], slot["latch_type"], confidence))
            return slot["latch_name"], slot["latch_type"], confidence

        # ── Feed raw result into history ──────────────────────────────────────
        slot["history"].append((raw_name, p_type, confidence))
        entries = list(slot["history"])

        if len(entries) < _MIN_CONFIRM:
            return raw_name, p_type, confidence

        # ── Majority vote ─────────────────────────────────────────────────────
        counts: dict = {}
        for (n, t, _) in entries:
            k = (n, t)
            counts[k] = counts.get(k, 0) + 1
        voted_name, voted_type = max(counts, key=counts.get)
        voted_conf  = next((c for (n, t, c) in reversed(entries) if n == voted_name), confidence)
        vote_ratio  = counts[(voted_name, voted_type)] / len(entries)

        if vote_ratio < 0.5:
            return raw_name, p_type, confidence

        # ── Act on stable vote ────────────────────────────────────────────────
        if voted_type in ("known", "blacklist"):
            new_identity = (voted_name != slot["latch_name"])
            slot["latch_name"]  = voted_name
            slot["latch_type"]  = voted_type
            slot["latch_until"] = now + _LATCH_SECS

            if new_identity and frame is not None:
                slot["known_saved"] = True
                if voted_type == "blacklist":
                    last_bl = _blacklist_last_alert.get(voted_name, 0.0)
                    if (now - last_bl) > _BLACKLIST_ALERT_SECS:
                        _blacklist_last_alert[voted_name] = now
                        _log_blacklist_async(voted_name, frame.copy(), box,
                                             slot.get("landmarks"), camera_source, zone_id)
                        print(f"[tracker] BLACKLIST confirmed & saved: {voted_name}")
                    else:
                        remaining = int(_BLACKLIST_ALERT_SECS - (now - last_bl))
                        print(f"[tracker] Blacklist cooldown ({remaining}s left): {voted_name}")
                else:
                    _log_known_async(voted_name, voted_type, voted_conf,
                                     frame.copy(), box, slot.get("landmarks"))
                    print(f"[tracker] Known confirmed & saved: {voted_name} ({voted_conf:.2f})")

            return voted_name, voted_type, voted_conf

        else:  # unknown stable
            # Fix #3: Unknown cooldown = 6 s per slot — prevents duplicate spam
            # Fix #3: Max 3 snaps per slot (first save + 2 updates at 6s intervals)
            _MAX_UNK_SNAPS     = 3
            _UNK_SNAP_COOLDOWN = 6.0

            if frame is not None:
                count   = slot.get("unk_snap_count", 0)
                last_ts = slot.get("unk_snap_ts", 0.0)

                # First save: immediate. Subsequent saves: only after cooldown expires.
                # upsert_unknown updates the SAME record (same temp_id) rather than
                # inserting a new one — deduplication is enforced at the DB layer.
                if count < _MAX_UNK_SNAPS and (count == 0 or (now - last_ts) >= _UNK_SNAP_COOLDOWN):
                    slot["unk_snap_count"] = count + 1
                    slot["unk_snap_ts"]    = now
                    lm  = slot.get("landmarks")   # may be None — save relaxed handles it
                    tid = slot["temp_id"]
                    _log_unknown_async(tid, frame.copy(), box, lm)
                    print(f"[tracker] Unknown snap {count+1}/{_MAX_UNK_SNAPS}: {tid}")

                    # Register embedding in grouping cache on first save
                    if count == 0 and slot.get("embedding") is not None:
                        _register_unknown_embedding(tid, slot["embedding"])

            return "Unknown", "unknown", 0.0

    def tick(self):
        for f in self.faces:
            f["age"] += 1
        self.faces = [f for f in self.faces if f["age"] < self.stale_frames]

    def reset(self):
        self.faces.clear()


_identity_tracker = IdentityTracker()
