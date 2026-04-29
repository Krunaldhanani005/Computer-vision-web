"""
face_engine/matcher.py — cosine distance matching against stored embeddings.
"""

import numpy as np
from models.face_recognition import arcface


def find_best_match(
    embedding: np.ndarray,
    encodings: list,
    names: list,
    threshold: float,
) -> tuple:
    """
    Find the closest stored encoding within cosine-distance threshold.

    Returns:
        (name, distance)  — if a match is found within threshold
        (None, None)      — if encodings is empty or no match
    """
    if not encodings:
        return None, None
    sims      = arcface.compute_similarities(encodings, embedding)
    distances = [1.0 - s if s != -1.0 else 2.0 for s in sims]
    best_idx  = int(np.argmin(distances))
    if distances[best_idx] <= threshold:
        return names[best_idx], distances[best_idx]
    return None, None
