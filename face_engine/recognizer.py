"""
face_engine/recognizer.py — ArcFace alignment + embedding extraction.

NOTE: Do NOT apply normalize_face (CLAHE/denoise) before get_embedding.
      Preprocessing breaks the ArcFace embedding space when comparing
      CCTV frames against enrollment photos (different lighting domains).
      arcface.get_embedding() handles its own (x/127.5 - 1) normalisation.
"""

import numpy as np
from models.face_recognition import arcface


def align_and_embed(frame: np.ndarray, landmarks) -> np.ndarray | None:
    """
    Align a detected face using 5-point SCRFD landmarks and return a
    512-d L2-normalised ArcFace embedding, or None on failure.
    """
    aligned = arcface.align_face(frame, landmarks)
    if aligned is None:
        return None
    return arcface.get_embedding(aligned)
