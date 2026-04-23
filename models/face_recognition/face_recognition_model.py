import cv2
import numpy as np
import face_recognition
from .face_detector import get_faces_dnn   # relative import — same package

# In-memory training store (cleared on page load)
known_encodings  = []
known_names      = []
DISTANCE_THRESHOLD = 0.55


def train_model(files, name):
    global known_encodings, known_names

    for file in files:
        data  = file.read()
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[DEBUG] train_model: Failed to decode image for {name}")
            continue

        faces = get_faces_dnn(image)
        print(f"[DEBUG] train_model: Number of faces detected = {len(faces)}")
        if faces:
            rgb_image      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
            encodings      = face_recognition.face_encodings(rgb_image, face_locations)
            for enc in encodings:
                known_encodings.append(enc)
                known_names.append(name)
                print(f"[DEBUG] train_model: Saved encoding for {name}. Encoding sample: {enc[:5]}...")

    print(f"[face_recognition] Active encodings: {len(known_encodings)}")

    if not known_encodings:
        return False, "No valid faces found. Please upload clear photos."
    return True, f"Trained '{name}' with {len(known_encodings)} encoding(s)."


def clear_model():
    """Reset in-memory training data."""
    global known_encodings, known_names
    known_encodings.clear()
    known_names.clear()
    _identity_tracker.reset()


def recognize(frame, face_box):
    """
    Accepts a full BGR frame and face_box (x, y, w, h).
    Returns e.g. "Alice (92%)" or "Unknown".
    """
    if frame is None or frame.size == 0:
        return "Unknown"

    x, y, w, h = face_box
    rgb_face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_location = [(y, x + w, y + h, x)]
    encodings = face_recognition.face_encodings(rgb_face, face_location)

    print(f"[DEBUG] recognize: Number of faces detected=1. Encoding generated={bool(encodings)}")

    if encodings and known_encodings:
        print(f"[DEBUG] recognize: Encoding value sample: {encodings[0][:5]}...")
        distances  = face_recognition.face_distance(known_encodings, encodings[0])
        print(f"[DEBUG] recognize: Distance scores: {distances}")
        best       = int(np.argmin(distances))
        if distances[best] < DISTANCE_THRESHOLD:
            conf = int((1 - distances[best]) * 100)
            matched_name = known_names[best]
            print(f"[DEBUG] recognize: Matched name: {matched_name} with conf: {conf}%")
            return f"{matched_name} ({conf}%)"
        else:
            print(f"[DEBUG] recognize: Matched name: None (best distance {distances[best]} >= {DISTANCE_THRESHOLD})")

    return "Unknown"

class IdentityTracker:
    def __init__(self, history_len=3, stale_frames=8):
        self.history_len = history_len
        self.stale_frames = stale_frames
        self.faces = []

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
        return inter / (union + 1e-5)

    def update(self, x, y, w, h, name):
        best_iou = 0
        best_idx = -1
        for i, face in enumerate(self.faces):
            iou = self._iou((x,y,w,h), face['box'])
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx != -1 and best_iou > 0.3:
            self.faces[best_idx]['box'] = (x,y,w,h)
            self.faces[best_idx]['age'] = 0
            if name != "Unknown":
                self.faces[best_idx]['history'].append(name)
        else:
            from collections import deque
            hist = deque(maxlen=self.history_len)
            if name != "Unknown":
                hist.append(name)
            self.faces.append({'box': (x,y,w,h), 'history': hist, 'age': 0})
            best_idx = len(self.faces) - 1

        hist = self.faces[best_idx]['history']
        if not hist:
            return "Unknown"

        # Majority vote
        labels = list(hist)
        stable_label = max(set(labels), key=labels.count)
        return stable_label

    def tick(self):
        for face in self.faces:
            face['age'] += 1
        self.faces = [f for f in self.faces if f['age'] < self.stale_frames]

    def reset(self):
        self.faces.clear()

_identity_tracker = IdentityTracker()
