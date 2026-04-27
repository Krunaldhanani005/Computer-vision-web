import cv2
import numpy as np
import onnxruntime as ort
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ARCFACE_MODEL = os.path.join(_BASE_DIR, "weights", "w600k_r50.onnx")

_arcface_session = None
if os.path.exists(_ARCFACE_MODEL):
    _arcface_session = ort.InferenceSession(_ARCFACE_MODEL, providers=['CPUExecutionProvider'])
    print("[ArcFace] Loaded ONNX model ✓")
else:
    print(f"[ArcFace] WARNING: Model not found at {_ARCFACE_MODEL}")

# Standard ArcFace reference landmarks for 112x112
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(img: np.ndarray, landmarks: list) -> np.ndarray:
    """
    Align face crop to 112x112 using 5 landmarks.
    landmarks should be: [right_eye_x, right_eye_y, left_eye_x, left_eye_y, nose_x, nose_y, right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y]
    """
    src = np.array(landmarks, dtype=np.float32).reshape(5, 2)
    tform, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_DST, method=cv2.LMEDS)
    if tform is None:
        return None
    aligned = cv2.warpAffine(img, tform, (112, 112), borderValue=0.0)
    return aligned

def get_embedding(aligned_face: np.ndarray) -> np.ndarray:
    """
    Get 512-d ArcFace embedding for an aligned 112x112 face crop.
    """
    if _arcface_session is None or aligned_face is None:
        return None

    # ArcFace preprocessing: (img / 127.5) - 1.0, RGB, NCHW
    rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(rgb, 1.0/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False)
    
    outputs = _arcface_session.run(None, {_arcface_session.get_inputs()[0].name: blob})
    embedding = outputs[0][0]
    
    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute Cosine similarity between two L2-normalized embeddings.
    Returns value between -1.0 and 1.0. Higher is better.
    """
    return float(np.dot(emb1, emb2))

def compute_similarities(known_embeddings: list, query_emb: np.ndarray) -> list:
    """
    Compute Cosine similarities against a list of embeddings.
    Ignores embeddings that do not match the query dimension (e.g. legacy dlib 128-d).
    Returns list of similarities corresponding to known_embeddings. If dimension mismatch,
    similarity is set to -1.0 (minimum possible).
    """
    similarities = []
    query_shape = query_emb.shape
    for emb in known_embeddings:
        if emb.shape != query_shape:
            similarities.append(-1.0)  # Incompatible embedding
        else:
            similarities.append(compute_similarity(emb, query_emb))
    return similarities
