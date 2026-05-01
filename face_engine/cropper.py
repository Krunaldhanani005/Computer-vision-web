"""
face_engine/cropper.py — canonical face crop with consistent padding.
Single source of truth for FACE_PAD so FR and RA save identical crops.
"""

import numpy as np

FACE_PAD = 0.40  # 40% expansion on each side


def crop_face(frame: np.ndarray, x: int, y: int, w: int, h: int,
              pad: float = FACE_PAD) -> np.ndarray:
    """
    Return a padded, clamped face crop from frame.
    Coordinates are explicitly clamped to frame boundaries.
    Returns empty array (size==0) if the resulting crop is invalid.
    """
    fh, fw = frame.shape[:2]
    pw = int(w * pad)
    ph = int(h * pad)
    x1 = max(0, x - pw)
    y1 = max(0, y - ph)
    x2 = min(fw, x + w + pw)
    y2 = min(fh, y + h + ph)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return crop
