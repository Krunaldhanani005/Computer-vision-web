"""
camera_manager.py  — Production-grade shared camera manager

Features:
    - Single background reader thread shared across all model streams
    - Bounded frame buffer (drops stale frames to prevent backlog)
    - RTSP low-latency optimisation (CAP_PROP_BUFFERSIZE = 1)
    - Automatic reconnect on RTSP failure (up to MAX_RECONNECT_ATTEMPTS)
    - Webcam and CCTV (RTSP/HTTP) support
"""

import cv2
import os
import threading
import time


# Reconnect settings for CCTV/RTSP streams
MAX_RECONNECT_ATTEMPTS = 10
RECONNECT_DELAY_SEC    = 2.0    # reduced from 3.0 — faster recovery on RTSP drop


class CameraManager:
    def __init__(self):
        self._cam              = None
        self._cam_lock         = threading.Lock()
        self._latest_frame     = None
        self._frame_lock       = threading.Lock()   # separate lock for frame
        self._running          = False
        self._thread           = None
        self._current_source   = None
        self._is_rtsp          = False              # True when source is a URL

    # ── Internal reader loop ──────────────────────────────────────────────────

    def _reader_loop(self):
        """
        Background thread: continuously reads frames and stores the latest.
        For RTSP streams, attempts automatic reconnection on failure.
        """
        consecutive_failures = 0

        while self._running:
            # ── Grab frame ───────────────────────────────────────────────────
            with self._cam_lock:
                if self._cam is None or not self._cam.isOpened():
                    ret, frame = False, None
                else:
                    ret, frame = self._cam.read()

            if ret and frame is not None:
                consecutive_failures = 0
                with self._frame_lock:
                    self._latest_frame = frame
                # Minimal sleep — let CPU breathe without capping FPS too hard
                time.sleep(0.001)
            else:
                consecutive_failures += 1

                if not self._is_rtsp:
                    # Webcam EOF / device error → stop
                    print("[camera_manager] Webcam read failure — stopping.")
                    break

                # RTSP: attempt reconnect
                print(f"[camera_manager] RTSP read failure #{consecutive_failures} — reconnecting in {RECONNECT_DELAY_SEC}s …")
                time.sleep(RECONNECT_DELAY_SEC)

                if consecutive_failures >= MAX_RECONNECT_ATTEMPTS:
                    print(f"[camera_manager] Max reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) reached — stopping.")
                    break

                with self._cam_lock:
                    if self._cam is not None:
                        self._cam.release()
                    self._cam = self._open_capture(self._current_source)
                    if self._cam and self._cam.isOpened():
                        print("[camera_manager] RTSP reconnected ✓")
                        consecutive_failures = 0
                    else:
                        print("[camera_manager] RTSP reconnect failed, retrying…")

        self._running = False
        print("[camera_manager] Reader thread exited.")

    # ── Capture factory ───────────────────────────────────────────────────────

    def _open_capture(self, source) -> cv2.VideoCapture:
        """Open a VideoCapture with source-appropriate settings."""
        is_rtsp = isinstance(source, str) and not source.isdigit()

        if is_rtsp:
            # Force TCP transport for RTSP — more stable than UDP on LAN/WiFi.
            # Must be set before VideoCapture() opens the connection.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            # Reset env var so other non-RTSP captures are unaffected
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

        return cap

    # ── Public open / close API ───────────────────────────────────────────────

    def _open_camera(self, source) -> bool:
        """Open camera source and start reader thread."""
        self.close_camera()  # Clean up any existing capture first

        cap = self._open_capture(source)
        if not cap.isOpened():
            print(f"[camera_manager] Failed to open source: {source}")
            cap.release()
            return False

        with self._cam_lock:
            self._cam = cap

        self._current_source = source
        self._is_rtsp        = not (isinstance(source, int) or
                                    (isinstance(source, str) and source.isdigit()))
        self._latest_frame   = None
        self._running        = True
        self._thread         = threading.Thread(
            target=self._reader_loop, daemon=True, name="CamReader"
        )
        self._thread.start()

        # Wait up to 2 s for first frame (ensures camera is actually producing)
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with self._frame_lock:
                if self._latest_frame is not None:
                    print(f"[camera_manager] Camera ready ✓ source={source!r}")
                    return True
            time.sleep(0.05)

        print(f"[camera_manager] Camera opened but no frame received within 2 s: {source!r}")
        return True   # Return True anyway; stream may warm up shortly

    def open_webcam(self, index: int = 0) -> bool:
        return self._open_camera(index)

    def open_cctv(self, url: str) -> bool:
        return self._open_camera(url)

    def close_camera(self):
        """Stop reader thread and release capture."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._cam_lock:
            if self._cam is not None:
                self._cam.release()
                self._cam = None

        with self._frame_lock:
            self._latest_frame = None

        self._current_source = None
        self._is_rtsp        = False

    # ── Frame access ──────────────────────────────────────────────────────────

    def get_latest_frame(self):
        """Return the most recent frame (or None). Always a copy to avoid races."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def is_running(self) -> bool:
        return self._running


# Global singleton
camera_manager = CameraManager()
