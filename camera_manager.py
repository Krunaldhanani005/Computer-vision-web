import cv2
import threading
import time

class CameraManager:
    def __init__(self):
        self._cam = None
        self._cam_lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None
        self._current_source = None

    def _reader_loop(self):
        while self._running:
            with self._cam_lock:
                if self._cam is None or not self._cam.isOpened():
                    break
                ret, frame = self._cam.read()
            if ret:
                self._latest_frame = frame
            else:
                # If we lose connection to RTSP or video ends, we might want to handle it.
                # For now, just continue or sleep
                time.sleep(0.01)
                continue
            time.sleep(0.005) # ~200 fps ceiling

    def _open_camera(self, source):
        # Close existing camera if different source or just to be safe
        self.close_camera()

        with self._cam_lock:
            self._cam = cv2.VideoCapture(source)
            # Apply settings only for webcam (int) to avoid issues with RTSP streams
            if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
                self._cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not self._cam.isOpened():
                self._cam = None
                return False

        self._current_source = source
        self._latest_frame = None
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def open_webcam(self, index=0):
        return self._open_camera(index)

    def open_cctv(self, url):
        return self._open_camera(url)

    def close_camera(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        
        with self._cam_lock:
            if self._cam is not None:
                self._cam.release()
                self._cam = None
        
        self._latest_frame = None
        self._current_source = None

    def get_latest_frame(self):
        return self._latest_frame

    def is_running(self):
        return self._running

# Global singleton instance
camera_manager = CameraManager()
