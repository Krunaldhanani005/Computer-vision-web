"""
face_engine/cropper.py — canonical face crop with consistent padding.
Single source of truth for _FACE_PAD so FR and RA save identical crops.
"""

import numpy as np

FACE_PAD = 0.35  # 35% expansion on each side — matches FR pipeline


def crop_face(frame: np.ndarray, x: int, y: int, w: int, h: int,
              pad: float = FACE_PAD) -> np.ndarray:
    """
    Return a padded face crop from frame.
    Clamps to frame boundaries; returns the sub-array (not a copy).
    """
    fh, fw = frame.shape[:2]
    pw = int(w * pad)
    ph = int(h * pad)
    x1 = max(0, x - pw)
    y1 = max(0, y - ph)
    x2 = min(fw, x + w + pw)
    y2 = min(fh, y + h + ph)
    return frame[y1:y2, x1:x2]
