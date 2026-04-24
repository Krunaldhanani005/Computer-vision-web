"""
models/restricted_area/recognizer.py
──────────────────────────────────────
Face-to-known-persons matching using cached encodings.
Zero DB queries during inference.
"""

import numpy as np
import face_recognition

_DISTANCE_THRESHOLD = 0.55

# In-memory cache (populated once at camera start)
_names:     list[str]        = []
_encodings: list[np.ndarray] = []


def load_cache(names: list[str], encodings: list[np.ndarray]) -> None:
    """Replace in-memory cache with freshly loaded DB data."""
    global _names, _encodings
    _names     = names
    _encodings = encodings
    print(f"[restricted_area.recognizer] Cache loaded: {len(_names)} known person(s).")


def match(unknown_encoding: np.ndarray) -> tuple[str, bool]:
    """
    Compare *unknown_encoding* against the cached known persons.

    Returns:
        (name,  True)   — if a match is found below the distance threshold
        ("Unknown", False) — otherwise
    """
    if not _encodings:
        return "Unknown", False

    try:
        distances = face_recognition.face_distance(_encodings, unknown_encoding)
        best_idx  = int(np.argmin(distances))
        if distances[best_idx] < _DISTANCE_THRESHOLD:
            return _names[best_idx], True
    except Exception as e:
        print(f"[restricted_area.recognizer] match error: {e}")

    return "Unknown", False
