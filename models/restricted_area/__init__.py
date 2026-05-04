"""
models/restricted_area/__init__.py
────────────────────────────────────
Independent Restricted Area surveillance module.

Shared:  SCRFD-10G face detector + ArcFace model (via face_engine)
Separate: restricted_area_db collections (ra_events, ra_alerts, ra_snapshots)

Pipeline per frame:
    detect faces (SCRFD multiscale, min_size=20)
    → quality gates (conf, size, landmarks, sharpness)
    → zone check (face center inside RA polygon — loaded from ra_zones)
    → ArcFace embed
    → match against RA authorised persons (authorized)
    → match against FR blacklist (blacklist / critical alert)
    → if unknown  inside zone: store + alert
    → if blacklist inside zone: store + critical alert
"""

import os
import time
import uuid
import threading

import cv2
import numpy as np

# ── Shared face engine (detection + recognition; data stored separately) ────
from face_engine import detect_faces, crop_face, align_and_embed, find_best_match
from face_engine.cropper import FACE_PAD

# ── RA-specific database ─────────────────────────────────────────────────────
from .database import (
    load_all_ra_known_persons,
    load_ra_zone,
    upsert_ra_event,
    insert_ra_alert,
)

# ── RA snapshot directories ───────────────────────────────────────────────────
_RA_SNAP_ROOT    = os.path.join("app", "static", "restricted_area")
_RA_SNAP_UNKNOWN = os.path.join(_RA_SNAP_ROOT, "unknown")
_RA_SNAP_BL      = os.path.join(_RA_SNAP_ROOT, "blacklist")
for _d in [_RA_SNAP_ROOT, _RA_SNAP_UNKNOWN, _RA_SNAP_BL]:
    os.makedirs(_d, exist_ok=True)

# ── Quality gate parameters (aligned with FR app-level gates) ────────────────
_MIN_CONF        = 0.20   # SCRFD detection confidence (matches FR app gate)
_MIN_FACE_SIZE   = 20     # px (matches FR app gate — supports far/small CCTV faces)
_MIN_SHARPNESS   = 8.0    # Laplacian variance (matches FR app gate)

# ── Recognition thresholds (cosine distance, aligned with FR thresholds) ──────
_AUTH_THRESHOLD  = 0.45   # ≤ this → authorized (matches FR KNOWN_THRESHOLD)
_BL_THRESHOLD    = 0.40   # ≤ this → blacklist  (matches FR BLACKLIST_THRESHOLD)

# ── Alert cooldown per slot ───────────────────────────────────────────────────
_ALERT_COOLDOWN    = 30.0   # seconds between repeat unknown alerts
_BL_ALERT_COOLDOWN = 15.0   # shorter cooldown for blacklist (critical)

# ── Minimum consecutive detections before first alert (prevents single-frame FP) ──
_MIN_CONFIRM = 3

# ── In-memory RA authorised persons ──────────────────────────────────────────
_ra_names:     list = []
_ra_encodings: list = []

# ── Per-face tracker slots ────────────────────────────────────────────────────
_slots: list = []   # [{box, age, event_id, last_alert}]

# ── Alert count per event_id ──────────────────────────────────────────────────
_alert_counts: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def load_known_persons() -> None:
    """Load authorised RA persons into memory. Call at camera start."""
    global _ra_names, _ra_encodings, _slots, _alert_counts
    _ra_names, _ra_encodings = load_all_ra_known_persons()
    _slots        = []
    _alert_counts = {}
    print(f"[restricted_area] {len(_ra_names)} RA authorised encodings loaded ✓")


def process_frame(frame: np.ndarray,
                  camera_source: str = "webcam",
                  zone_id: str = "Default") -> list:
    """
    Run full RA pipeline on one BGR frame.
    Returns list of dicts: {box:(x1,y1,x2,y2), label, color, person_type}
    """
    results = []
    try:
        # Zone required — no zone → no detection (strict enforcement)
        zone_pts = load_ra_zone("restricted_default")
        if zone_pts is None or len(zone_pts) < 3:
            return []

        fh, fw = frame.shape[:2]

        # ── Detect all faces (multiscale: near + medium + far) ──────────────
        face_boxes = detect_faces(frame, min_size=_MIN_FACE_SIZE)

        for face_data in face_boxes:
            if len(face_data) < 5:
                continue

            x, y, w, h = face_data[0], face_data[1], face_data[2], face_data[3]
            landmarks  = face_data[4]
            det_conf   = float(face_data[5]) if len(face_data) >= 6 else 0.5

            # Gate 1: detection confidence
            if det_conf < _MIN_CONF:
                continue

            # Gate 2: face size (allows far faces >= 50px)
            if w < _MIN_FACE_SIZE or h < _MIN_FACE_SIZE:
                continue

            # Gate 3: valid landmarks (strongest false-positive filter)
            if landmarks is None:
                continue

            # Gate 4: aspect ratio (same range as FR — filters tilted/skewed detections)
            aspect = w / (h + 1e-5)
            if aspect < 0.4 or aspect > 2.2:
                continue

            # Gate 5: sharpness
            x1c = max(0, x);    y1c = max(0, y)
            x2c = min(fw, x+w); y2c = min(fh, y+h)
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue
            blur = float(cv2.Laplacian(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if blur < _MIN_SHARPNESS:
                continue

            # Gate 5: zone check — use RA zone points, NOT FR zone manager
            if not _is_in_zone(x, y, w, h, zone_pts, fw, fh):
                continue

            # ── ArcFace encode ───────────────────────────────────────────────
            embedding = align_and_embed(frame, landmarks)
            if embedding is None:
                continue

            # ── Match: RA authorised → FR blacklist → unknown ────────────────
            person_type, matched_name = _match(embedding)

            if person_type == "authorized":
                color = (0, 220, 80)
                label = matched_name
            elif person_type == "blacklist":
                color = (0, 0,220)
                label = f"⚠ BLACKLIST: {matched_name}"
            else:
                color = (0, 120, 255)
                label = "UNKNOWN ⚠"

            results.append({
                "box":         (x, y, x + w, y + h),
                "label":       label,
                "color":       color,
                "person_type": person_type,
            })

            # ── Fire alert for intruders ─────────────────────────────────────
            if person_type in ("unknown", "blacklist"):
                _handle_intruder(
                    frame, x, y, w, h,
                    camera_source, zone_id, person_type, matched_name
                )

        # Age-out stale slots
        _tick_slots()

    except Exception as e:
        print(f"[restricted_area] process_frame error: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _is_in_zone(x: int, y: int, w: int, h: int,
                zone_pts: list, fw: int, fh: int) -> bool:
    """Point-in-polygon test using face centre and RA zone points."""
    if not zone_pts or len(zone_pts) < 3:
        return False  # Strict mode: must be inside a valid zone
    cx = float(x + w / 2)
    cy = float(y + h / 2)
    pixel_pts = [(int(p["x"] * fw), int(p["y"] * fh)) for p in zone_pts]
    polygon = np.array([[px, py] for px, py in pixel_pts], dtype=np.int32)
    result = cv2.pointPolygonTest(
        polygon.reshape((-1, 1, 2)), (cx, cy), measureDist=False
    )
    return result >= 0


def _match(embedding: np.ndarray) -> tuple:
    """
    Returns (person_type, name):
        ("authorized", name)  — RA authorized person
        ("blacklist",  name)  — FR blacklist person (critical alert)
        ("unknown", "Unknown") — unrecognized intruder
    """
    # 1. Check RA authorized persons first
    name, _ = find_best_match(embedding, _ra_encodings, _ra_names, _AUTH_THRESHOLD)
    if name is not None:
        return "authorized", name

    # 2. Check FR blacklist encodings
    try:
        from models.face_recognition.face_recognition_model import (
            known_encodings, known_names, known_types,
        )
        bl_encs  = [e for e, t in zip(known_encodings, known_types) if t == "blacklist"]
        bl_names = [n for n, t in zip(known_names,    known_types) if t == "blacklist"]
        bl_name, _ = find_best_match(embedding, bl_encs, bl_names, _BL_THRESHOLD)
        if bl_name is not None:
            return "blacklist", bl_name
    except Exception as e:
        print(f"[ra._match] blacklist check error: {e}")

    return "unknown", "Unknown"


def _iou(bA, bB) -> float:
    xA = max(bA[0], bB[0]);  yA = max(bA[1], bB[1])
    xB = min(bA[0]+bA[2], bB[0]+bB[2])
    yB = min(bA[1]+bA[3], bB[1]+bB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    union = bA[2]*bA[3] + bB[2]*bB[3] - inter
    return inter / (union + 1e-5)


def _find_or_create_slot(x: int, y: int, w: int, h: int) -> int:
    """IoU-based slot matching — each unique face gets its own slot."""
    box = (x, y, w, h)
    best_iou, best_idx = 0.0, -1
    for i, slot in enumerate(_slots):
        iou = _iou(box, slot["box"])
        if iou > best_iou:
            best_iou, best_idx = iou, i

    if best_idx != -1 and best_iou > 0.20:
        _slots[best_idx]["box"] = box
        _slots[best_idx]["age"] = 0
        return best_idx

    # New slot — separate event_id per unknown person
    _slots.append({
        "box":           box,
        "age":           0,
        "event_id":      uuid.uuid4().hex[:10],
        "last_alert":    0.0,
        "confirm_count": 0,   # consecutive detections; must reach _MIN_CONFIRM before alerting
    })
    return len(_slots) - 1


def _handle_intruder(frame: np.ndarray, x: int, y: int, w: int, h: int,
                     camera_source: str, zone_id: str,
                     person_type: str, name: str = "Unknown"):
    """Per-slot cooldown + confirmation gate → fire alert for unknown/blacklist intruders."""
    idx  = _find_or_create_slot(x, y, w, h)
    slot = _slots[idx]
    now  = time.monotonic()

    # Require _MIN_CONFIRM consecutive detections before first alert (blocks single-frame FP)
    slot["confirm_count"] = slot.get("confirm_count", 0) + 1
    if slot["confirm_count"] < _MIN_CONFIRM:
        return

    cooldown = _BL_ALERT_COOLDOWN if person_type == "blacklist" else _ALERT_COOLDOWN
    if (now - slot["last_alert"]) < cooldown:
        return

    slot["last_alert"] = now
    evt = slot["event_id"]
    cnt = _alert_counts.get(evt, 0)
    max_snaps = 8 if person_type == "blacklist" else 5
    if cnt >= max_snaps:
        return
    _alert_counts[evt] = cnt + 1

    _fire_alert_async(
        frame.copy(), x, y, w, h,
        evt, camera_source, zone_id, person_type, name
    )


def _fire_alert_async(frame_copy: np.ndarray, x: int, y: int, w: int, h: int,
                      event_id: str, camera_source: str, zone_id: str,
                      person_type: str, name: str):
    """Save snapshot + write to RA DB in daemon thread."""
    def _worker():
        try:
            crop = crop_face(frame_copy, x, y, w, h)   # 35% pad via face_engine.FACE_PAD
            if crop.size == 0:
                return

            sub      = "blacklist" if person_type == "blacklist" else "unknown"
            snap_dir = _RA_SNAP_BL  if person_type == "blacklist" else _RA_SNAP_UNKNOWN
            fname    = f"ra_{sub}_{event_id}_{uuid.uuid4().hex[:6]}.jpg"
            fpath    = os.path.join(snap_dir, fname)
            snap_path = f"static/restricted_area/{sub}/{fname}"

            ok = cv2.imwrite(fpath, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                print(f"[restricted_area] imwrite failed: {fpath}")
                return

            upsert_ra_event(event_id, camera_source, zone_id, snap_path, person_type, name)
            insert_ra_alert(event_id, snap_path, camera_source, zone_id, person_type, name)
            print(f"[restricted_area] {person_type.upper()} alert → {snap_path}")
        except Exception as e:
            print(f"[restricted_area] alert worker error: {e}")

    threading.Thread(target=_worker, daemon=True).start()


def _tick_slots():
    """Age slots; remove stale ones (unseen > 25 frames)."""
    for s in _slots:
        s["age"] += 1
    _slots[:] = [s for s in _slots if s["age"] < 25]
