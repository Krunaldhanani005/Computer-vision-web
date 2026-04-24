print("Starting server... loading libraries (this may take a minute)")
import time
import threading
import cv2
from flask import Flask, Response, jsonify, render_template, request

# ── AI model imports (new modular structure) ──────────────────────────────────
from models.face_recognition import train_model, recognize, clear_model, get_faces_dnn, _identity_tracker
from models.emotion_detection.emotion_model import predict_emotion, face_cascade, smooth_emotion
from models.object_detection import detect_objects, draw_detections
import os
from models.plate_detection.plate_model import generate_plate_frames, stop_video_stream

print("Libraries loaded. Initialising Flask app...")
app = Flask(__name__)

# Temporary upload folder — files are deleted after processing
TEMP_FOLDER = "temp_uploads"
os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED CAMERA — one VideoCapture + one background reader thread
# ═══════════════════════════════════════════════════════════════════════════════
_shared_cam      = None
_shared_cam_lock = threading.Lock()
_latest_frame    = None
_cam_running     = False


def _cam_reader_loop():
    """Background thread: continuously grabs the newest frame from the webcam."""
    global _latest_frame, _cam_running
    while _cam_running:
        with _shared_cam_lock:
            if _shared_cam is None or not _shared_cam.isOpened():
                break
            ret, frame = _shared_cam.read()
        if ret:
            _latest_frame = frame
        time.sleep(0.005)   # ~200 fps ceiling — prevents CPU spin


def _open_shared_camera():
    """Open the camera and start the reader thread. Returns True on success."""
    global _shared_cam, _cam_running, _latest_frame
    with _shared_cam_lock:
        if _shared_cam is not None and _shared_cam.isOpened():
            return True
        _shared_cam = cv2.VideoCapture(0)
        _shared_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _shared_cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        _shared_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not _shared_cam.isOpened():
            _shared_cam = None
            return False
    _latest_frame = None
    _cam_running  = True
    threading.Thread(target=_cam_reader_loop, daemon=True).start()
    return True


def _close_shared_camera():
    """Stop the reader thread and release the webcam."""
    global _shared_cam, _cam_running, _latest_frame
    _cam_running = False
    time.sleep(0.05)   # let reader thread exit cleanly
    with _shared_cam_lock:
        if _shared_cam is not None:
            _shared_cam.release()
            _shared_cam = None
    _latest_frame = None


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════
is_fr_streaming      = False   # face-recognition stream
is_emotion_streaming = False   # emotion-detection stream
is_object_streaming  = False   # object-detection stream
is_restricted_streaming = False # restricted-area stream
fr_frame_counter     = 0
_emotion_frame_counter = 0
_object_frame_counter  = 0
restricted_frame_counter = 0


# ── Utility ───────────────────────────────────────────────────────────────────
def _padded_crop(frame, x, y, w, h, pad: float = 0.15):
    """
    Return the face region with proportional padding on all sides.
    Padding gives the model more context (forehead, chin, cheeks) which
    improves emotion recognition accuracy.
    """
    fh, fw   = frame.shape[:2]
    pad_w    = int(w * pad)
    pad_h    = int(h * pad)
    x1, y1   = max(0, x - pad_w),  max(0, y - pad_h)
    x2, y2   = min(fw, x + w + pad_w), min(fh, y + h + pad_h)
    return frame[y1:y2, x1:x2]


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — general
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    name  = request.form.get("name", "").strip()
    person_type = request.form.get("type", "known").strip()
    files = request.files.getlist("images")
    print(f"[train] name={name!r} type={person_type!r} files={len(files)}")
    if not name or not files:
        return "Provide name and at least 1 image", 400
    success, message = train_model(files, name, person_type)
    return "trained" if success else message, (200 if success else 400)


@app.route("/clear_training", methods=["POST"])
def clear_training():
    clear_model()
    return jsonify({"success": True})


# ═══════════════════════════════════════════════════════════════════════════════
#  FACE RECOGNITION STREAM
# ═══════════════════════════════════════════════════════════════════════════════
def generate_fr_frames():
    """MJPEG generator — DNN face detection + face-recognition overlay."""
    global is_fr_streaming, fr_frame_counter
    while is_fr_streaming:
        frame = _latest_frame
        if frame is None:
            time.sleep(0.01)
            continue
        frame = frame.copy()

        fr_frame_counter += 1
        if fr_frame_counter % 2 != 0:   # process every 2nd frame for smooth FPS
            time.sleep(0.01)
            continue

        scale       = 1.5
        search_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        faces       = get_faces_dnn(search_frame, smooth=True)

        for (x, y, w, h) in faces:
            ox = int(x / scale);  oy = int(y / scale)
            ow = int(w / scale);  oh = int(h / scale)

            raw_name, p_type = recognize(frame, (ox, oy, ow, oh))
            smooth_name, smooth_type = _identity_tracker.update(ox, oy, ow, oh, raw_name, p_type)
            
            if smooth_name != "Unknown":
                if smooth_type == "blacklist":
                    color = (0, 0, 255) # Red for blacklist
                    smooth_name = f"ALERT: {smooth_name}"
                else:
                    color = (0, 255, 0) # Green for known
            else:
                color = (0, 165, 255) # Orange for unknown

            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), color, 2)
            cv2.putText(frame, smooth_name, (ox, oy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        _identity_tracker.tick()

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/fr_video_feed")
def fr_video_feed():
    return Response(generate_fr_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start_fr_camera", methods=["POST"])
def start_fr_camera():
    global is_fr_streaming
    
    # Reload encodings when camera starts (per user requirement)
    from models.face_recognition.face_recognition_model import load_encodings_from_db
    load_encodings_from_db()
    
    if not _open_shared_camera():
        return jsonify({"success": False, "message": "Cannot open webcam."}), 500
    is_fr_streaming = True
    return jsonify({"success": True})


@app.route("/stop_fr_camera", methods=["POST"])
def stop_fr_camera():
    global is_fr_streaming
    is_fr_streaming = False
    if not is_emotion_streaming and not is_object_streaming and not is_restricted_streaming:
        _close_shared_camera()
    return jsonify({"success": True})


# ═══════════════════════════════════════════════════════════════════════════════
#  EMOTION DETECTION STREAM
# ═══════════════════════════════════════════════════════════════════════════════
def generate_emotion_frames():
    global is_emotion_streaming, _emotion_frame_counter

    while is_emotion_streaming:
        frame = _latest_frame
        if frame is None:
            time.sleep(0.01)
            continue
        frame = frame.copy()

        _emotion_frame_counter += 1
        if _emotion_frame_counter % 2 != 0:
            time.sleep(0.01)
            continue

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            if w < 50 or h < 50:
                continue

            crop = _padded_crop(frame, x, y, w, h, pad=0.15)
            
            emotion, conf = predict_emotion(crop)
            
            if emotion != "Unknown":
                emotion = smooth_emotion(emotion)
                
            display = f"{emotion} ({int(conf*100)}%)" if emotion != "Unknown" else "Unknown"
            color = (0, 255, 0) if emotion != "Unknown" else (140, 140, 140)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - th - 14), (x + tw + 6, y), (0, 0, 0), -1)
            cv2.putText(frame, display, (x + 3, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

@app.route("/start_emotion")
def start_emotion():
    return Response(generate_emotion_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_emotion_camera", methods=["POST"])
def start_emotion_camera():
    global is_emotion_streaming
    if not _open_shared_camera():
        return jsonify({"success": False, "message": "Cannot open webcam."}), 500
    is_emotion_streaming = True
    return jsonify({"success": True})

@app.route("/stop_emotion_camera", methods=["POST"])
def stop_emotion_camera():
    global is_emotion_streaming
    is_emotion_streaming = False
    if not is_fr_streaming and not is_object_streaming and not is_restricted_streaming:
        _close_shared_camera()
    return jsonify({"success": True})

# ═══════════════════════════════════════════════════════════════════════════════
#  OBJECT DETECTION STREAM
# ═══════════════════════════════════════════════════════════════════════════════
def generate_object_frames():
    """MJPEG generator — YOLOv8 object detection overlay (filtered classes only)."""
    global is_object_streaming, _object_frame_counter
    while is_object_streaming:
        frame = _latest_frame
        if frame is None:
            time.sleep(0.01)
            continue
        frame = frame.copy()

        _object_frame_counter += 1
        if _object_frame_counter % 2 != 0:
            time.sleep(0.01)
            continue

        detections = detect_objects(frame)
        draw_detections(frame, detections)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/start_object_detection")
def start_object_detection():
    return Response(generate_object_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start_object_camera", methods=["POST"])
def start_object_camera():
    global is_object_streaming
    if not _open_shared_camera():
        return jsonify({"success": False, "message": "Cannot open webcam."}), 500
    is_object_streaming = True
    return jsonify({"success": True})


@app.route("/stop_object_camera", methods=["POST"])
def stop_object_camera():
    global is_object_streaming
    is_object_streaming = False
    if not is_fr_streaming and not is_emotion_streaming and not is_restricted_streaming:
        _close_shared_camera()
    return jsonify({"success": True})

# ═══════════════════════════════════════════════════════════════════════════════
#  RESTRICTED AREA DETECTION STREAM
# ═══════════════════════════════════════════════════════════════════════════════
def generate_restricted_frames():
    """MJPEG generator — full Restricted Area pipeline (YOLO → face → match)."""
    global is_restricted_streaming, restricted_frame_counter
    from models.restricted_area import process_frame

    while is_restricted_streaming:
        frame = _latest_frame
        if frame is None:
            time.sleep(0.01)
            continue
        frame = frame.copy()

        restricted_frame_counter += 1
        if restricted_frame_counter % 3 != 0:   # run pipeline at ~10 FPS
            time.sleep(0.01)
            continue

        frame = process_frame(frame)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

@app.route("/restricted_video_feed")
def restricted_video_feed():
    return Response(generate_restricted_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_restricted_camera", methods=["POST"])
def start_restricted_camera():
    global is_restricted_streaming
    # Load fresh known-person encodings from MongoDB into RAM cache
    from models.restricted_area import load_known_persons
    load_known_persons()

    if not _open_shared_camera():
        return jsonify({"success": False, "message": "Cannot open webcam."}), 500
    is_restricted_streaming = True
    return jsonify({"success": True})

@app.route("/stop_restricted_camera", methods=["POST"])
def stop_restricted_camera():
    global is_restricted_streaming
    is_restricted_streaming = False
    if not is_fr_streaming and not is_emotion_streaming and not is_object_streaming:
        _close_shared_camera()
    return jsonify({"success": True})


@app.route("/add_known_person", methods=["POST"])
def add_known_person():
    """
    Register an authorised person for Restricted Area detection.
    Expects multipart/form-data:
        name   : str
        images : 1+ image files
    Stores face encodings in restricted_area_db.known_persons.
    Images are NEVER saved to disk.
    """
    from models.restricted_area.face_handler import extract_encoding_from_image
    from models.restricted_area.database    import insert_known_person

    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "message": "Name is required."}), 400

    files = request.files.getlist("images")
    if len(files) < 1:
        return jsonify({"success": False, "message": "Upload at least 1 image."}), 400

    success_count = 0
    skipped_count = 0

    for f in files:
        img_bytes = f.read()
        if not img_bytes:
            skipped_count += 1
            continue

        encoding = extract_encoding_from_image(img_bytes)
        if encoding is None:
            skipped_count += 1
            continue

        if insert_known_person(name, encoding):
            success_count += 1
        else:
            skipped_count += 1

    if success_count == 0:
        return jsonify({
            "success": False,
            "message": f"No valid faces found in {len(files)} image(s). Upload clear, front-facing photos."
        }), 422

    return jsonify({
        "success": True,
        "message": f"Registered '{name}' with {success_count} encoding(s). ({skipped_count} skipped)"
    })


@app.route("/get_alerts")
def get_alerts():
    """Return the 20 most-recent intrusion alerts as JSON for the UI table."""
    from models.restricted_area.database import get_recent_alerts
    return jsonify(get_recent_alerts())


# ═══════════════════════════════════════════════════════════════════════════════
#  LICENSE PLATE DETECTION STREAM
# ═══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _delete_file_later(path, delay: int = 30):
    """Safety net: delete *path* after *delay* seconds even if streaming aborted."""
    def _worker():
        time.sleep(delay)
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"[cleanup] Safety-deleted temp file: {path}")
        except Exception as e:
            print(f"[cleanup] Could not delete {path}: {e}")
    threading.Thread(target=_worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# VEHICLE DETECTION — UPLOAD & STREAM
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/upload_video", methods=["POST"])
def upload_video():
    print("[upload_video] Request received")

    if 'video' not in request.files:
        return jsonify({"error": "No file attached"}), 400

    file = request.files['video']

    if not file or file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # ── Security: extension whitelist ────────────────────────────────────────
    allowed_extensions = {'.mp4', '.avi', '.mov'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Allowed: .mp4, .avi, .mov"}), 400

    # ── Security: file-size guard ────────────────────────────────────────────
    if request.content_length and request.content_length > MAX_FILE_SIZE:
        return jsonify({"error": "File too large (max 20 MB)"}), 413

    # ── Save to TEMP folder — NOT a permanent upload directory ───────────────
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    temp_path = os.path.join(TEMP_FOLDER, file.filename)
    file.save(temp_path)
    print(f"[upload_video] Saved temporarily at: {temp_path}")

    # Start safety-net deletion thread (fires 30 s after upload)
    _delete_file_later(temp_path, delay=30)

    return jsonify({"filename": file.filename})


@app.route('/video_feed/<filename>')
def plate_video_feed(filename):
    """Stream vehicle-detection MJPEG frames; temp file is deleted after streaming."""
    temp_path = os.path.join(TEMP_FOLDER, filename)
    return Response(
        generate_plate_frames(temp_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stop_video', methods=['POST'])
def stop_video():
    stop_video_stream()
    return jsonify({"status": "stopped"})



if __name__ == "__main__":
    print("Server starting on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
