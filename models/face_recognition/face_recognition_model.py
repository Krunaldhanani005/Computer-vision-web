import cv2
import numpy as np
import face_recognition
import datetime
from threading import Thread
from pymongo import MongoClient
from .face_detector import get_faces_dnn

# MongoDB Setup
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['cv_project']
    faces_collection = db['faces']
    logs_collection = db['logs']
except Exception as e:
    print(f"[face_recognition] MongoDB connection error: {e}")

# In-memory store
known_encodings  = []
known_names      = []
known_types      = []
DISTANCE_THRESHOLD = 0.55

# Logging control
last_logged = {} # { "name": last_datetime }
LOG_COOLDOWN = 10 # seconds

def load_encodings_from_db():
    global known_encodings, known_names, known_types
    known_encodings.clear()
    known_names.clear()
    known_types.clear()
    
    try:
        count = 0
        for face in faces_collection.find():
            if 'encoding' in face:
                known_encodings.append(np.array(face['encoding']))
                known_names.append(face.get('name', 'Unknown'))
                known_types.append(face.get('type', 'known'))
                count += 1
        print(f"[face_recognition] Loaded {count} encodings from MongoDB.")
    except Exception as e:
        print(f"[face_recognition] Error loading from DB: {e}")

# Load encodings once at startup
load_encodings_from_db()

def _log_detection_async(name, person_type):
    def worker():
        try:
            logs_collection.insert_one({
                "name": name,
                "type": person_type,
                "timestamp": datetime.datetime.utcnow()
            })
        except Exception as e:
            print(f"[face_recognition] Error logging detection: {e}")
    Thread(target=worker, daemon=True).start()

def train_model(files, name, person_type="known"):
    global known_encodings, known_names, known_types

    success_count = 0
    for file in files:
        data  = file.read()
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[DEBUG] train_model: Failed to decode image for {name}")
            continue

        faces = get_faces_dnn(image)
        if faces:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
            encodings = face_recognition.face_encodings(rgb_image, face_locations)
            for enc in encodings:
                try:
                    # Save to DB
                    faces_collection.insert_one({
                        "name": name,
                        "type": person_type,
                        "encoding": enc.tolist(),
                        "created_at": datetime.datetime.utcnow()
                    })
                    
                    # Append to memory
                    known_encodings.append(enc)
                    known_names.append(name)
                    known_types.append(person_type)
                    success_count += 1
                    print(f"[DEBUG] train_model: Saved encoding for {name} ({person_type}) to MongoDB.")
                except Exception as e:
                    print(f"[face_recognition] DB insert error: {e}")

    print(f"[face_recognition] Active encodings: {len(known_encodings)}")

    if success_count == 0:
        return False, "No valid faces found. Please upload clear photos."
    return True, f"Trained '{name}' ({person_type}) with {success_count} encoding(s)."

def clear_model():
    """Reset in-memory training data. Note: Does not delete from MongoDB!"""
    global known_encodings, known_names, known_types
    known_encodings.clear()
    known_names.clear()
    known_types.clear()
    _identity_tracker.reset()
    # Optionally reload from DB again to restore state
    load_encodings_from_db()

def recognize(frame, face_box):
    """
    Accepts a full BGR frame and face_box (x, y, w, h).
    Returns (formatted_name, person_type).
    """
    if frame is None or frame.size == 0:
        return "Unknown", "unknown"

    x, y, w, h = face_box
    rgb_face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_location = [(y, x + w, y + h, x)]
    encodings = face_recognition.face_encodings(rgb_face, face_location)

    if encodings and known_encodings:
        distances  = face_recognition.face_distance(known_encodings, encodings[0])
        best       = int(np.argmin(distances))
        if distances[best] < DISTANCE_THRESHOLD:
            conf = int((1 - distances[best]) * 100)
            matched_name = known_names[best]
            matched_type = known_types[best]
            
            # Logging logic
            now = datetime.datetime.now()
            if matched_name not in last_logged or (now - last_logged[matched_name]).total_seconds() > LOG_COOLDOWN:
                last_logged[matched_name] = now
                _log_detection_async(matched_name, matched_type)
                
            return f"{matched_name} ({conf}%)", matched_type

    return "Unknown", "unknown"

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

    def update(self, x, y, w, h, name, p_type="unknown"):
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
                self.faces[best_idx]['history'].append((name, p_type))
        else:
            from collections import deque
            hist = deque(maxlen=self.history_len)
            if name != "Unknown":
                hist.append((name, p_type))
            self.faces.append({'box': (x,y,w,h), 'history': hist, 'age': 0})
            best_idx = len(self.faces) - 1

        hist = self.faces[best_idx]['history']
        if not hist:
            return "Unknown", "unknown"

        # Majority vote
        labels = list(hist)
        stable_tuple = max(set(labels), key=labels.count)
        return stable_tuple

    def tick(self):
        for face in self.faces:
            face['age'] += 1
        self.faces = [f for f in self.faces if f['age'] < self.stale_frames]

    def reset(self):
        self.faces.clear()

_identity_tracker = IdentityTracker()
