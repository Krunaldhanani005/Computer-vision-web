"""
models/restricted_area/tracker.py
───────────────────────────────────
Lightweight IoU tracker — assigns stable IDs to detected persons
across frames and enforces per-ID alert cooldowns.
"""

import time

_COOLDOWN_SEC  = 10     # seconds between alerts for the same track ID
_MAX_STALE     = 15     # frames a track can go unseen before removal (at ~10 FPS ≈ 1.5s)
_IOU_THRESHOLD = 0.35   # min overlap to match old ↔ new box

_next_id    = 1
_tracks     = {}   # id -> {"box", "age", "last_alert"}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1)) - inter
    return inter / (union + 1e-5)


def update(boxes: list[tuple]) -> list[dict]:
    """
    Match new bounding boxes to existing tracks via IoU.

    Returns list of dicts:
        { "id": int, "box": (x1,y1,x2,y2), "cooldown_ok": bool }
    """
    global _next_id, _tracks

    matched_old_ids = set()
    results = []

    for box in boxes:
        best_id, best_score = None, _IOU_THRESHOLD
        for tid, track in _tracks.items():
            score = _iou(track["box"], box)
            if score > best_score:
                best_score = score
                best_id = tid

        if best_id is not None:
            _tracks[best_id]["box"] = box
            _tracks[best_id]["age"] = 0
            matched_old_ids.add(best_id)
            tid = best_id
        else:
            tid = _next_id
            _next_id += 1
            _tracks[tid] = {"box": box, "age": 0, "last_alert": 0}

        elapsed = time.time() - _tracks[tid]["last_alert"]
        results.append({
            "id":          tid,
            "box":         box,
            "cooldown_ok": elapsed > _COOLDOWN_SEC,
        })

    # Age out unseen tracks
    for tid in list(_tracks.keys()):
        if tid not in matched_old_ids:
            _tracks[tid]["age"] += 1
            if _tracks[tid]["age"] > _MAX_STALE:
                del _tracks[tid]

    return results


def mark_alerted(track_id: int) -> None:
    """Record that an alert was just fired for this track."""
    if track_id in _tracks:
        _tracks[track_id]["last_alert"] = time.time()


def reset() -> None:
    global _next_id, _tracks
    _next_id = 1
    _tracks  = {}
