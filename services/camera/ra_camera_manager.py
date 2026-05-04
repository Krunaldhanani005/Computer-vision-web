"""
ra_camera_manager.py — Dedicated camera manager for Restricted Area module.

Completely SEPARATE from the shared FR camera_manager.
This eliminates device-lock races when FR and RA are started/stopped.
"""
from services.camera.camera_manager import CameraManager

# Independent singleton — not shared with FR or object detection
ra_camera_manager = CameraManager()
