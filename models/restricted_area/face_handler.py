"""
models/restricted_area/face_handler.py
────────────────────────────────────────
Face extraction and ArcFace encoding for RA enrollment.
Uses face_engine (SCRFD-10G detector + ArcFace model).
Data stored in restricted_area_db (separate from fr_surveillance_db).
"""

import cv2
import numpy as np

from face_engine import detect_faces, align_and_embed


def extract_encoding_from_image(img_bytes: bytes) -> np.ndarray | None:
    """
    Accept raw image bytes (Flask file upload).
    Detect best face, align, return 512-d ArcFace embedding.
    Returns None if no valid face found.
    """
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            print("[ra.face_handler] Could not decode image.")
            return None

        faces = detect_faces(bgr, min_size=50)
        if not faces:
            print("[ra.face_handler] No face detected in image.")
            return None

        # Pick face with highest confidence (index 5)
        best_face = max(faces, key=lambda f: f[5] if len(f) >= 6 else 0)
        landmarks = best_face[4]
        if landmarks is None:
            print("[ra.face_handler] No valid landmarks.")
            return None

        embedding = align_and_embed(bgr, landmarks)
        if embedding is None:
            print("[ra.face_handler] ArcFace embedding failed.")
            return None

        print("[ra.face_handler] Encoding extracted ✓ (512-d ArcFace)")
        return embedding

    except Exception as e:
        print(f"[ra.face_handler] Error: {e}")
        return None
