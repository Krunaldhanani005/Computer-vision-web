"""
face_engine — shared face detection, cropping, embedding, and matching.

Both FR and RA pipelines import from here so parameter changes propagate
to both systems from a single location.
"""

from .detector  import detect_faces
from .cropper   import crop_face, FACE_PAD
from .recognizer import align_and_embed
from .matcher   import find_best_match
