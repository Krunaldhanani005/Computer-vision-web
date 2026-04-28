"""
face_recognition_model.py  — Production-grade FR pipeline

Pipeline:
    frame → detect (YOLOv8n-face) → encode (dlib 128-d) → compare
    → classify (known / unknown / blacklist)
    → cooldown check → snapshot save → DB update (async)

Cooldown:
    known / blacklist : 10 s
    unknown           : 10 s de-duplication per tracker slot
"""

import cv2
import numpy as np
from . import arcface
import datetime
import os
import uuid
from threading import Thread
from collections import deque

from .face_detector import get_faces_dnn

from .fr_database import (
    insert_face,
    get_all_faces,
    delete_all_faces,
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

# ── Recognition parameters ────────────────────────────────────────────────────
# Calibrated for real-world CCTV. Cosine distance (1 - similarity)
DISTANCE_THRESHOLD = 0.48    # <= this → Known match. Allow tuning if needed.

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

# Absolute minimums — only reject completely un-usable crops
_MIN_SAVE_W = 40    # px — very lenient; CCTV faces can be small
_MIN_SAVE_H = 40


def _sharpness_score(gray_crop: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Used for ranking only, not rejection."""
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


_MIN_SAVE_W = 30
_MIN_SAVE_H = 30
_MIN_SHARPNESS = 10.0

def _validate_face_for_save(frame: np.ndarray, box) -> tuple:
    """
    Quality gate for face snapshots.
    Returns: (is_valid, sharpness, reason)
    """
    x, y, w, h = box
    fh, fw = frame.shape[:2]

    if w <= 0 or h <= 0:
        return False, 0.0, f"zero_size ({w}x{h})"

    # Minimum face size gate
    if w < _MIN_SAVE_W or h < _MIN_SAVE_H:
        return False, 0.0, f"too_small ({w}x{h})"

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    raw = frame[y1:y2, x1:x2]
    if raw.size == 0:
        return False, 0.0, "empty_crop"

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return True, sharpness, "ok"


# ══════════════════════════════════════════════════════════════════════════════
#  SNAPSHOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Per-identity frame buffer: key → {'frames': [(sharpness, frame, box)], 'ts': t}
_FRAME_BUFFER: dict = {}
_FRAME_BUFFER_MAX_FRAMES = 5
_FRAME_BUFFER_MAX_SECS   = 3.0


def _collect_best_frame(key: str, frame: np.ndarray, box, sharpness: float) -> tuple:
    """Buffer removed. Immediate return for detection-first approach."""
    return frame.copy(), box


def _crop_face(frame: np.ndarray, box, landmarks=None):
    """
    Crop the face region. If landmarks are provided, align the face using ArcFace.
    Resize to 160x160 as requested.
    """
    if landmarks is not None:
        aligned = arcface.align_face(frame, landmarks)
        if aligned is not None:
            # Aligned is 112x112, resize to 160x160
            return cv2.resize(aligned, (160, 160), interpolation=cv2.INTER_CUBIC)
            
    # Fallback to unaligned crop if no landmarks
    x, y, w, h = box
    fh, fw = frame.shape[:2]
    
    pad_w = int(w * 0.15)
    pad_top = int(h * 0.20)
    pad_bot = int(h * 0.10)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_top)
    x2 = min(fw, x + w + pad_w)
    y2 = min(fh, y + h + pad_bot)

    crop = frame[y1:y2, x1:x2]
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
    """Quality-gated save for Known / Blacklist (strict)."""
    try:
        valid, sharpness, reason = _validate_face_for_save(frame, box)
        if not valid:
            print(f"[snapshot] Rejected ({reason}) label={label}")
            return ""
        crop = _crop_face(frame, box, landmarks)
        if crop is None:
            print(f"[snapshot] _crop_face returned None for label={label}")
            return ""
        return _write_image(crop, folder or _SNAP_DIR, label)
    except Exception as e:
        print(f"[snapshot] Error: {e}")
        return ""


def _save_snapshot_relaxed(frame: np.ndarray, box, label: str, folder: str = None, landmarks: list = None) -> str:
    """Relaxed save for Unknown — never rejects on quality, only on absolute size."""
    try:
        x, y, w, h = box
        fh, fw = frame.shape[:2]
        if w < 25 or h < 25:
            print(f"[snapshot-unk] Skip: too tiny ({w}x{h})")
            return ""
        crop = _crop_face(frame, box, landmarks)
        if crop is None:
            # Hard fallback: raw region resized to 160x160
            raw = frame[max(0,y):min(fh,y+h), max(0,x):min(fw,x+w)]
            if raw.size == 0:
                print(f"[snapshot-unk] Empty raw crop for label={label}")
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


def _log_blacklist_async(name: str, frame: np.ndarray, box, landmarks=None, camera_source="webcam", zone_id="default"):
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
    Align face, generate ArcFace embedding, and compare against known_encodings.
    face_data: (x, y, w, h, landmarks) OR (x, y, w, h, landmarks, conf)
    Accepts both 5-element and 6-element tuples (detector returns 6 with conf).
    """
    if frame is None or frame.size == 0:
        return "Unknown", "unknown", 0.0, 2.0

    # ── FIX: accept 5 OR 6 element tuples (detector returns 6: x,y,w,h,lm,conf) ──
    if len(face_data) < 5:
        print(f"[recognize] ERROR: face_data too short (len={len(face_data)}), need >=5")
        return "Unknown", "unknown", 0.0, 2.0

    x, y, w, h, landmarks = face_data[0], face_data[1], face_data[2], face_data[3], face_data[4]

    # ── Gate 1: Minimum face size (50px allows far faces) ──────────────────
    if w < 50 or h < 50:
        print(f"[recognize] Skip: face too small ({w}x{h} < 50px)")
        return "Unknown", "unknown", 0.0, 2.0

    fh, fw = frame.shape[:2]
    face_crop_raw = frame[max(0, y):min(fh, y+h), max(0, x):min(fw, x+w)]
    if face_crop_raw.size == 0:
        return "Unknown", "unknown", 0.0, 2.0

    # ── Gate 2: Sharpness check (far faces have lower natural sharpness) ──────
    gray_crop = cv2.cvtColor(face_crop_raw, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    if blur_score < 8.0:
        print(f"[recognize] Skip: too blurry (sharpness={blur_score:.1f} < 8)")
        return "Unknown", "unknown", 0.0, 2.0

    # ── Align and Encode with ArcFace ─────────────────────────────────────────
    aligned = arcface.align_face(frame, landmarks)
    if aligned is None:
        print(f"[recognize] WARNING: alignment failed for face {w}x{h} (landmarks={landmarks is not None})")
        return "Unknown", "unknown", 0.0, 2.0

    # Pass raw aligned face — no CLAHE/denoising (breaks ArcFace embedding space)
    new_enc = arcface.get_embedding(aligned)
    if new_enc is None:
        print(f"[recognize] ERROR: ArcFace embedding returned None")
        return "Unknown", "unknown", 0.0, 2.0

    if not known_encodings:
        print(f"[recognize] WARNING: No known encodings in memory — run training first!")
        return "Unknown", "unknown", 0.0, 2.0

    # ── Compare against ALL stored encodings (exhaustive best-match) ──────────
    similarities = arcface.compute_similarities(known_encodings, new_enc)
    distances = [1.0 - sim if sim != -1.0 else 2.0 for sim in similarities]

    # Per-person best (minimum distance across all embeddings for that person)
    person_best: dict = {}
    for i, dist in enumerate(distances):
        nm = known_names[i]
        pt = known_types[i]
        if nm not in person_best or dist < person_best[nm][0]:
            person_best[nm] = (float(dist), pt)

    # Separate by type — Priority: Blacklist > Known
    blacklist_best = None
    known_best = None

    for nm, (dist, pt) in person_best.items():
        if pt == "blacklist":
            if blacklist_best is None or dist < blacklist_best[1]:
                blacklist_best = (nm, dist, pt)
        else:
            if known_best is None or dist < known_best[1]:
                known_best = (nm, dist, pt)

    # ── Debug log ─────────────────────────────────────────────────────────────
    best_overall_name = "N/A"
    best_overall_dist = 2.0
    if person_best:
        best_overall_name, (best_overall_dist, _) = min(
            person_best.items(), key=lambda kv: kv[1][0]
        )
    print(
        f"[recognize] Face {w}x{h} | sharpness={blur_score:.1f} | "
        f"best_match={best_overall_name} | dist={best_overall_dist:.3f} | "
        f"threshold={DISTANCE_THRESHOLD} | encodings_in_db={len(known_encodings)}"
    )

    # 1. Blacklist check first
    if blacklist_best and blacklist_best[1] <= DISTANCE_THRESHOLD:
        best_name, best_dist, best_type = blacklist_best
        conf_score = max(0.0, 1.0 - best_dist)
        print(f"[recognize] RESULT: BLACKLIST | name={best_name} | dist={best_dist:.3f} | conf={conf_score:.2f}")
        return best_name, best_type, conf_score, best_dist

    # 2. Known check second
    if known_best and known_best[1] <= DISTANCE_THRESHOLD:
        best_name, best_dist, best_type = known_best
        conf_score = max(0.0, 1.0 - best_dist)
        print(f"[recognize] RESULT: KNOWN | name={best_name} | dist={best_dist:.3f} | conf={conf_score:.2f}")
        return best_name, best_type, conf_score, best_dist

    # 3. Else Unknown
    best_dist_final = min([d for _, (d, _) in person_best.items()] + [2.0])
    print(f"[recognize] RESULT: Unknown | best_dist={best_dist_final:.3f} | threshold={DISTANCE_THRESHOLD}")
    return "Unknown", "unknown", 0.0, best_dist_final


# ══════════════════════════════════════════════════════════════════════════════
#  IDENTITY TRACKER  (time-based latch + majority vote + stable saves)
# ══════════════════════════════════════════════════════════════════════════════

import time as _time

_LATCH_SECS  = 5.0   # Once Known confirmed, hold identity for this many seconds
_MIN_CONFIRM = 5     # Frames of agreement needed before saving to DB (higher = fewer false positives)


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
               confidence: float = 0.0, frame=None, landmarks=None, camera_source="webcam", zone_id="default"):
        now = _time.monotonic()
        box = (x, y, w, h)

        # ── Find matching slot by IoU ────────────────────────────────────────────
        best_iou, best_idx = 0.0, -1
        for i, f in enumerate(self.faces):
            iou = self._iou(box, f["box"])
            if iou > best_iou:
                best_iou, best_idx = iou, i

        if best_idx != -1 and best_iou > 0.20:
            slot = self.faces[best_idx]
            slot["box"] = box
            slot["age"] = 0
            if landmarks: slot["landmarks"] = landmarks
        else:
            # New slot
            tid  = uuid.uuid4().hex[:8]
            slot = {
                "box":          box,
                "landmarks":    landmarks,
                "age":          0,
                "temp_id":      tid,
                "history":      deque(maxlen=self.history_len),
                "latch_name":   None,
                "latch_type":   None,
                "latch_until":  0.0,
                "unk_saved":    False,
                "known_saved":  False,
            }
            self.faces.append(slot)
            best_idx = len(self.faces) - 1

        slot = self.faces[best_idx]

        # ── LATCH: if Known identity confirmed and still fresh, return it ────────
        if now < slot["latch_until"] and slot["latch_name"]:
            # Keep feeding into history to maintain vote, but don't flip identity
            slot["history"].append((slot["latch_name"], slot["latch_type"], confidence))
            return slot["latch_name"], slot["latch_type"], confidence

        # ── Feed raw recognition result into history ───────────────────────────
        slot["history"].append((raw_name, p_type, confidence))
        entries = list(slot["history"])

        # Not enough history yet — show label but don’t save
        if len(entries) < _MIN_CONFIRM:
            return raw_name, p_type, confidence

        # ── Majority vote ───────────────────────────────────────────────────
        counts: dict = {}
        for (n, t, _) in entries:
            k = (n, t)
            counts[k] = counts.get(k, 0) + 1
        voted_name, voted_type = max(counts, key=counts.get)
        voted_conf  = next((c for (n, t, c) in reversed(entries) if n == voted_name), confidence)
        vote_ratio  = counts[(voted_name, voted_type)] / len(entries)

        # Require 50%+ agreement before acting on the vote
        if vote_ratio < 0.5:
            return raw_name, p_type, confidence

        # ── Act on stable vote ─────────────────────────────────────────────────
        if voted_type in ("known", "blacklist"):
            new_identity = (voted_name != slot["latch_name"])
            # Set or refresh latch
            slot["latch_name"]  = voted_name
            slot["latch_type"]  = voted_type
            slot["latch_until"] = now + _LATCH_SECS
            slot["unk_saved"]   = False  # reset unknown flag if identity confirmed

            # Save to DB only when identity first confirmed (or changed)
            if new_identity and frame is not None:
                slot["known_saved"] = True
                if voted_type == "blacklist":
                    # Per-person cooldown — prevents alert flood from multiple slots
                    last_bl = _blacklist_last_alert.get(voted_name, 0.0)
                    if (now - last_bl) > _BLACKLIST_ALERT_SECS:
                        _blacklist_last_alert[voted_name] = now
                        _log_blacklist_async(voted_name, frame.copy(), box, slot.get("landmarks"), camera_source, zone_id)
                        print(f"[tracker] ⚠ Blacklist CONFIRMED & saved: {voted_name}")
                    else:
                        print(f"[tracker] ⚠ Blacklist confirmed (cooldown active, {int(_BLACKLIST_ALERT_SECS - (now - last_bl))}s left): {voted_name}")
                else:
                    _log_known_async(voted_name, voted_type, voted_conf, frame.copy(), box, slot.get("landmarks"))
                    print(f"[tracker] ✓ Known CONFIRMED & saved: {voted_name} ({voted_conf:.2f})")

            return voted_name, voted_type, voted_conf

        else:  # unknown stable
            # Save unknown only ONCE per slot AND only when landmarks are valid
            # (no landmarks = likely a false-positive background detection)
            if not slot["unk_saved"] and frame is not None:
                lm = slot.get("landmarks")
                if lm is None:
                    print(f"[tracker] Skip Unknown save: no landmarks (false positive guard) id={slot['temp_id']}")
                    return "Unknown", "unknown", 0.0
                slot["unk_saved"] = True
                _log_unknown_async(slot["temp_id"], frame.copy(), box, lm)
                print(f"[tracker] ? Unknown CONFIRMED & saved: {slot['temp_id']}")
            return "Unknown", "unknown", 0.0

    def tick(self):
        for f in self.faces:
            f["age"] += 1
        self.faces = [f for f in self.faces if f["age"] < self.stale_frames]

    def reset(self):
        self.faces.clear()


_identity_tracker = IdentityTracker()
