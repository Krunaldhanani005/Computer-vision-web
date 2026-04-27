import cv2
import numpy as np

def align_face(img, landmarks):
    src = np.array(landmarks, dtype=np.float32).reshape(5, 2)
    # arcface standard landmarks for 112x112
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    
    tform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if tform is None:
        return None
    aligned = cv2.warpAffine(img, tform, (112, 112), borderValue=0.0)
    return aligned

print("Alignment script ok")
