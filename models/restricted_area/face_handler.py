"""
models/restricted_area/face_handler.py
───────────────────────────────────────
Face extraction and 128-d encoding generation.
Completely independent from models/face_recognition/.
"""

import cv2
import numpy as np
import face_recognition   # dlib-backed library


def extract_encoding_from_image(img_bytes: bytes) -> np.ndarray | None:
    """
    Accept raw image bytes (from a Flask file upload),
    detect the largest face, and return its 128-d encoding.

    Returns:
        np.ndarray of shape (128,)  — on success
        None                        — if no face found / image invalid
    """
    try:
        # 1. Decode bytes → numpy BGR image
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            print("[face_handler] Could not decode image bytes.")
            return None

        # 2. Resize for faster processing (max dimension 800px)
        h, w  = bgr.shape[:2]
        scale = min(800 / max(h, w, 1), 1.0)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

        # 3. Convert BGR → RGB (face_recognition requires RGB)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 4. Locate all face bounding boxes
        face_locations = face_recognition.face_locations(rgb, model="hog")
        if not face_locations:
            print("[face_handler] No face detected in image.")
            return None

        # 5. If multiple faces, pick the largest (biggest bounding box area)
        if len(face_locations) > 1:
            face_locations = [
                max(face_locations,
                    key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
            ]

        # 6. Generate 128-d encoding
        encodings = face_recognition.face_encodings(rgb, face_locations)
        if not encodings:
            print("[face_handler] Encoding extraction failed.")
            return None

        return encodings[0]   # np.ndarray (128,)

    except Exception as e:
        print(f"[face_handler] Unexpected error: {e}")
        return None
