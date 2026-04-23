import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.npy")

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

emotion_history = deque(maxlen=5)

def smooth_emotion(emotion):
    emotion_history.append(emotion)
    return max(set(emotion_history), key=emotion_history.count)

def predict_emotion(face_img):
    if face_img is None or face_img.size == 0:
        return "Unknown", 0.0

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))

    face = face / 255.0
    face = face.reshape(1, 48, 48, 1)

    preds = model.predict(face, verbose=0)[0]

    idx = np.argmax(preds)
    confidence = float(preds[idx])

    emotion = labels[idx]

    return emotion, confidence
