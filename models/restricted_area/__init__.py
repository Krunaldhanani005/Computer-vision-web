"""
models/restricted_area/__init__.py
────────────────────────────────────
Controller — orchestrates the full per-frame pipeline:
  detector  →  face_handler  →  recognizer  →  tracker  →  database

Exposes two public functions used by app.py:
    load_known_persons()   — call once at camera start
    process_frame(frame)   — call each processed frame; returns annotated frame
"""

import os
import time
import threading
import cv2
import numpy as np
import face_recognition

from . import detector, face_handler, recognizer, tracker
from .database import load_all_known_persons, log_alert

# ── Constants ─────────────────────────────────────────────────────────────────
_ALERTS_DIR        = os.path.join("static", "alerts")
_DISTANCE_THRESH   = 0.55   # face match cutoff
_DEDUP_COOLDOWN    = 10     # seconds — reject duplicate alerts for same face
_DEDUP_MATCH_DIST  = 0.45   # encoding distance to consider "same person"

# Colours
_COLOR_KNOWN   = (0, 220, 80)     # green
_COLOR_UNKNOWN = (0, 0, 220)      # red  (BGR)

os.makedirs(_ALERTS_DIR, exist_ok=True)

# ── Encoding-based dedup cache ────────────────────────────────────────────────
# Each entry: { "encoding": np.ndarray, "time": float }
_recent_alerts: list[dict] = []


# ── Public initialisation ─────────────────────────────────────────────────────

def load_known_persons() -> None:
    """Load authorised encodings from MongoDB into recognizer's RAM cache."""
    global _recent_alerts
    names, encodings = load_all_known_persons()
    recognizer.load_cache(names, encodings)
    tracker.reset()
    _recent_alerts = []   # clear dedup cache on fresh start


# ── Per-frame processing ──────────────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Run the full Restricted Area pipeline on one BGR frame.
    Returns the annotated frame (never raises).
    """
    try:
        # 1. Detect persons (YOLO)
        person_boxes = detector.get_persons(frame)
        if not person_boxes:
            return frame

        # 2. Update tracker → get stable IDs + cooldown flags
        tracks = tracker.update(person_boxes)

        for track in tracks:
            tid        = track["id"]
            box        = track["box"]
            cooldown_ok = track["cooldown_ok"]
            x1, y1, x2, y2 = box

            # 3. Extract face encoding from person crop
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            encoding = _get_encoding_from_crop(person_crop)

            if encoding is None:
                # No face visible — draw a neutral grey box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                continue

            # 4. Match against cached known persons
            name, is_known = recognizer.match(encoding)

            # 5. Draw annotated box
            color = _COLOR_KNOWN if is_known else _COLOR_UNKNOWN
            label = name if is_known else "UNKNOWN — ALERT"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # 6. Fire alert if unknown + tracker cooldown OK + encoding dedup OK
            if not is_known and cooldown_ok:
                if not _is_duplicate_alert(encoding):
                    tracker.mark_alerted(tid)
                    _register_alert_encoding(encoding)
                    _fire_alert_async(frame.copy(), x1, y1, x2, y2)

    except Exception as e:
        print(f"[restricted_area] process_frame error: {e}")

    return frame


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_encoding_from_crop(bgr_crop: np.ndarray) -> np.ndarray | None:
    """Convert a BGR person crop → 128-d face encoding, or None."""
    try:
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        if not locs:
            return None
        encs = face_recognition.face_encodings(rgb, locs)
        return encs[0] if encs else None
    except Exception:
        return None


def _is_duplicate_alert(encoding: np.ndarray) -> bool:
    """
    Check if this face encoding was already logged recently.
    Compares against _recent_alerts using distance + time window.
    Also garbage-collects expired entries.
    """
    global _recent_alerts
    now = time.time()

    # Purge stale entries (older than cooldown)
    _recent_alerts = [e for e in _recent_alerts if (now - e["time"]) < _DEDUP_COOLDOWN]

    # Compare encoding against all recent alerts
    for entry in _recent_alerts:
        dist = float(np.linalg.norm(encoding - entry["encoding"]))
        if dist < _DEDUP_MATCH_DIST:
            return True   # same person, still within cooldown → duplicate

    return False


def _register_alert_encoding(encoding: np.ndarray) -> None:
    """Record this encoding so future frames can detect duplicates."""
    _recent_alerts.append({"encoding": encoding.copy(), "time": time.time()})


def _fire_alert_async(frame_copy: np.ndarray,
                      x1: int, y1: int, x2: int, y2: int) -> None:
    """Save snapshot + log to MongoDB in a daemon thread."""
    def _worker():
        try:
            ts         = int(time.time())
            filename   = f"intruder_{ts}.jpg"
            filepath   = os.path.join(_ALERTS_DIR, filename)
            db_path    = f"static/alerts/{filename}"

            # Save only the person crop (not full frame)
            crop = frame_copy[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                cv2.imwrite(filepath, crop)

            log_alert(db_path, status="unknown")
            print(f"[restricted_area] Alert saved → {filepath}")
        except Exception as e:
            print(f"[restricted_area] Alert worker error: {e}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
