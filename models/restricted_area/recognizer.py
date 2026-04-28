"""
models/restricted_area/recognizer.py
──────────────────────────────────────
ArcFace cosine-distance matching for RA authorised persons.
Uses shared ArcFace model; data (encodings) come from restricted_area_db only.
"""

import numpy as np
from models.face_recognition import arcface

_RA_DISTANCE_THRESHOLD = 0.50  # cosine distance; <= this → authorised

_names:     list = []
_encodings: list = []


def load_cache(names: list, encodings: list) -> None:
    global _names, _encodings
    _names     = names
    _encodings = encodings
    print(f"[ra.recognizer] Loaded {len(_names)} ArcFace RA encoding(s).")


def match(unknown_embedding: np.ndarray) -> tuple:
    """
    Returns (name, is_known):
        (name,      True)  — authorised person matched
        ("Unknown", False) — no match → intruder
    """
    if not _encodings:
        return "Unknown", False
    try:
        similarities = arcface.compute_similarities(_encodings, unknown_embedding)
        distances    = [1.0 - s if s != -1.0 else 2.0 for s in similarities]
        best_idx     = int(np.argmin(distances))
        if distances[best_idx] <= _RA_DISTANCE_THRESHOLD:
            return _names[best_idx], True
    except Exception as e:
        print(f"[ra.recognizer] match error: {e}")
    return "Unknown", False
