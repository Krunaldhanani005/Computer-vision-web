"""
models/restricted_area/recognizer.py
──────────────────────────────────────
ArcFace cosine-distance matching for RA authorised persons.
Uses face_engine.find_best_match; data (encodings) come from restricted_area_db only.
"""

import numpy as np
from face_engine import find_best_match

_RA_DISTANCE_THRESHOLD = 0.45  # cosine distance ≤ this → authorised (matches FR KNOWN_THRESHOLD)

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
    try:
        name, _ = find_best_match(unknown_embedding, _encodings, _names, _RA_DISTANCE_THRESHOLD)
        if name is not None:
            return name, True
    except Exception as e:
        print(f"[ra.recognizer] match error: {e}")
    return "Unknown", False
