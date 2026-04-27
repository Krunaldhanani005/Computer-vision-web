import cv2
import numpy as np
import onnxruntime as ort
import os

yunet_path = "models/face_recognition/weights/face_detection_yunet.onnx"
arcface_path = "models/face_recognition/weights/w600k_r50.onnx"

print(f"YuNet exists: {os.path.exists(yunet_path)}")
print(f"ArcFace exists: {os.path.exists(arcface_path)}")

# Load YuNet
detector = cv2.FaceDetectorYN_create(yunet_path, "", (320, 320), 0.5, 0.3, 5000)
print("YuNet loaded")

# Load ArcFace
session = ort.InferenceSession(arcface_path, providers=['CPUExecutionProvider'])
print("ArcFace loaded")
