"""
face_engine/detector.py — canonical face detection entry point.
Wraps SCRFD-10G multiscale detector from models.face_recognition.
"""

from models.face_recognition.face_detector import detect_faces_multiscale


def detect_faces(frame, min_size: int = 20):
    """
    Detect faces in frame using SCRFD-10G (multiscale: normal + 1.5× upscale).

    Returns list of (x, y, w, h, landmarks, score) tuples.
    min_size: smallest face side in pixels to accept.
    """
    return detect_faces_multiscale(frame, min_size=min_size)
