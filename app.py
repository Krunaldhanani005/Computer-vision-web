print("Starting server... loading libraries (this may take a minute)")
import time
import threading
import cv2
import numpy as np
import concurrent.futures
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

# ── AI model imports (modular structure) ──────────────────────────────────────
from models.face_recognition import (
    train_model, recognize, clear_model, get_faces_dnn,
    _identity_tracker, reset_tracking_state, load_encodings_from_db
)
from models.emotion_detection.emotion_model import predict_emotion, face_cascade, smooth_emotion
from models.object_detection import detect_objects, draw_detections
import os
from urllib.parse import quote as _url_quote
from dotenv import load_dotenv
load_dotenv()
from models.plate_detection.plate_model import generate_plate_frames, stop_video_stream

print("Libraries loaded. Initialising Flask app...")
app = Flask(__name__)

# Temporary upload folder — files are deleted after processing
TEMP_FOLDER = "temp_uploads"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Snapshot folder
SNAPSHOT_FOLDER = os.path.join("static", "fr_logs")
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED CAMERA — managed by camera_manager
# ═══════════════════════════════════════════════════════════════════════════════
from camera_manager import camera_manager
from zone_manager import zone_manager

camera_source_config = {"type": "webcam", "url": None}


def _build_cctv_url():
    """Build authenticated RTSP/HTTP URL from .env credentials."""
    raw_url  = os.getenv("CCTV_URL", "").strip()
    username = os.getenv("CCTV_USERNAME", "").strip()
    password = os.getenv("CCTV_PASSWORD", "").strip()

    if not raw_url:
        return None, "CCTV_URL is not configured in .env"

    if username and password:
        creds = f"{_url_quote(username, safe='')}:{_url_quote(password, safe='')}"
        for scheme in ("rtsp://", "http://", "https://"):
            if raw_url.startswith(scheme):
                raw_url = f"{scheme}{creds}@{raw_url[len(scheme):]}"
                break

    return raw_url, None


def stop_all_models():
    """Ensure only one model is active at a time."""
    global is_fr_streaming, is_emotion_streaming, is_object_streaming, is_restricted_streaming
    is_fr_streaming      = False
    is_emotion_streaming = False
    is_object_streaming  = False
    is_restricted_streaming = False


def _handle_source_switch(data):
    """Apply camera source from request data."""
    if not data:
        return None
    source_type = data.get("source", "webcam")

    if source_type == "cctv":
        url, err = _build_cctv_url()
        if err:
            return err
    else:
        url = None

    if camera_source_config["type"] != source_type or camera_source_config["url"] != url:
        _close_shared_camera()
        camera_source_config["type"] = source_type
        camera_source_config["url"]  = url

    return None


def _open_shared_camera():
    if camera_manager.is_running():
        return True
    if camera_source_config["type"] == "webcam":
        return camera_manager.open_webcam(0)
    else:
        return camera_manager.open_cctv(camera_source_config["url"])


def _close_shared_camera():
    camera_manager.close_camera()


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════
is_fr_streaming         = False
is_emotion_streaming    = False
is_object_streaming     = False
is_restricted_streaming = False
fr_frame_counter        = 0
_emotion_frame_counter  = 0
_object_frame_counter   = 0
restricted_frame_counter = 0

# Async recognition state
fr_executor    = concurrent.futures.ThreadPoolExecutor(max_workers=1)
fr_future      = None
fr_last_results: list = []


# ── Utility ───────────────────────────────────────────────────────────────────
def _padded_crop(frame, x, y, w, h, pad: float = 0.15):
    fh, fw = frame.shape[:2]
    pad_w  = int(w * pad)
    pad_h  = int(h * pad)
    x1, y1 = max(0, x - pad_w),  max(0, y - pad_h)
    x2, y2 = min(fw, x + w + pad_w), min(fh, y + h + pad_h)
    return frame[y1:y2, x1:x2]


# Debug overlay info shared between async task and draw loop
_fr_debug_persons: list = []   # [(x1,y1,x2,y2), ...] in original frame coords


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — general pages
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live-demo")
def live_demo():
    return render_template("live_demo.html")

@app.route("/face-recognition")
def face_recognition_page():
    return render_template("face_recognition.html")

@app.route("/emotion-detection")
def emotion_detection_page():
    return render_template("emotion_detection.html")

@app.route("/object-detection")
def object_detection_page():
    return render_template("object_detection.html")

@app.route("/restricted-area")
def restricted_area_page():
    return render_template("restricted_area.html")

@app.route("/report")
def report_page():
    return render_template("report.html")

@app.route("/fr-report")
def fr_report_page():
    return render_template("fr_report.html")

@app.route("/details")
def details_page():
    return render_template("details.html")

@app.route("/license")
def license_page():
    return render_template("license.html")

@app.route("/set_camera_source", methods=["POST"])
def set_camera_source():
    data        = request.json
    source_type = data.get("type", "webcam")
    url         = data.get("url", None)
    _close_shared_camera()
    camera_source_config["type"] = source_type
    camera_source_config["url"]  = url
    success = _open_shared_camera()
    return jsonify({"success": success})


# ── Snapshot serving ──────────────────────────────────────────────────────────
@app.route("/static/fr_logs/<filename>")
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOT_FOLDER, filename)

@app.route("/static/fr_reports/<filename>")
def serve_report_snapshot(filename):
    return send_from_directory(os.path.join("static", "fr_reports"), filename)


# ═══════════════════════════════════════════════════════════════════════════════
#  FR REPORT DASHBOARD  API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/report/known")
def api_report_known():
    from models.face_recognition.fr_reports_db import get_known_dashboard
    search   = request.args.get("search", "").strip()
    sort_by  = request.args.get("sort", "last_seen")
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 24))
    docs, total = get_known_dashboard(search, sort_by, page, per_page)
    return jsonify({"data": docs, "total": total, "page": page, "per_page": per_page})


@app.route("/api/report/known/stats")
def api_report_known_stats():
    from models.face_recognition.fr_reports_db import get_known_stats
    return jsonify(get_known_stats())


@app.route("/api/report/unknown")
def api_report_unknown():
    from models.face_recognition.fr_reports_db import get_unknown_dashboard
    search   = request.args.get("search", "").strip()
    sort_by  = request.args.get("sort", "last_seen")
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    docs, total = get_unknown_dashboard(search, sort_by, page, per_page)
    return jsonify({"data": docs, "total": total, "page": page, "per_page": per_page})

@app.route("/api/report/unknown/delete_all", methods=["POST"])
def api_report_unknown_delete_all():
    from models.face_recognition.fr_reports_db import delete_all_unknown_reports
    from models.face_recognition.fr_database import delete_all_unknown_logs
    success1 = delete_all_unknown_reports()
    success2 = delete_all_unknown_logs()
    if success1 or success2:
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Failed to delete reports"}), 500



@app.route("/api/report/unknown/stats")
def api_report_unknown_stats():
    from models.face_recognition.fr_reports_db import get_unknown_stats
    return jsonify(get_unknown_stats())


@app.route("/api/report/blacklist")
def api_report_blacklist():
    from models.face_recognition.fr_reports_db import get_blacklist_dashboard
    search      = request.args.get("search", "").strip()
    date_filter = request.args.get("date", "").strip()
    page        = int(request.args.get("page", 1))
    per_page    = int(request.args.get("per_page", 50))
    docs, total = get_blacklist_dashboard(search, date_filter, page, per_page)
    return jsonify({"data": docs, "total": total, "page": page, "per_page": per_page})


@app.route("/api/report/blacklist/stats")
def api_report_blacklist_stats():
    from models.face_recognition.fr_reports_db import get_blacklist_stats
    return jsonify(get_blacklist_stats())


@app.route("/api/report/export/known")
def api_export_known():
    from models.face_recognition.fr_reports_db import export_known_csv
    from flask import make_response
    csv_data = export_known_csv()
    resp = make_response(csv_data)
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=known_persons_report.csv"
    return resp


@app.route("/api/report/export/blacklist")
def api_export_blacklist():
    from models.face_recognition.fr_reports_db import export_blacklist_csv
    from flask import make_response
    csv_data = export_blacklist_csv()
    resp = make_response(csv_data)
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=blacklist_alerts_report.csv"
    return resp

# ═══════════════════════════════════════════════════════════════════════════════
#  POLYGON ZONE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/save_zone", methods=["POST"])
def save_zone():
    global fr_last_results
    data      = request.json
    zone_data = data.get("zone", None)

    reset_tracking_state()
    fr_last_results = []

    if zone_data is None:
        zone_manager.clear_zone()
        print("[zone] Zone cleared")
        return jsonify({"success": True})

    # zone_data is a list of {x, y} normalised points
    if isinstance(zone_data, list):
        success = zone_manager.save_zone(zone_data)
        print(f"[zone] Polygon zone saved: {len(zone_data)} points → {success}")
        return jsonify({"success": success})

    # Legacy rectangle format fallback (should not occur with new UI)
    if isinstance(zone_data, dict) and "points" in zone_data:
        success = zone_manager.save_zone(zone_data["points"])
        return jsonify({"success": success})

    return jsonify({"success": False, "error": "Invalid zone format"}), 400


@app.route("/get_zone", methods=["GET"])
def get_zone():
    pts = zone_manager.load_zone()
    return jsonify({"zone": pts})


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING ROUTE
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/train", methods=["POST"])
def train():
    name        = request.form.get("name", "").strip()
    person_type = request.form.get("type", "known").strip()
    files       = request.files.getlist("images")
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
#  DASHBOARD API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/fr/recognized")
def api_recognized():
    from models.face_recognition.fr_database import get_recognized_dashboard
    return jsonify(get_recognized_dashboard())


@app.route("/api/fr/unknown")
def api_unknown():
    from models.face_recognition.fr_database import get_unknown_dashboard
    return jsonify(get_unknown_dashboard())


@app.route("/api/fr/alerts")
def api_alerts():
    from models.face_recognition.fr_database import get_alerts_dashboard
    return jsonify(get_alerts_dashboard())


# ═══════════════════════════════════════════════════════════════════════════════
#  FACE RECOGNITION STREAM  (polygon zone-aware)
# ═══════════════════════════════════════════════════════════════════════════════

def _fr_async_task(small_frame: np.ndarray, orig_frame: np.ndarray,
                    frame_w: int, frame_h: int):
    """
    Background recognition task.

    small_frame : 640-px downscaled copy  (YOLO person detection — fast)
    orig_frame  : full-resolution copy    (face detection + recognition — quality)
    frame_w/h   : orig_frame dimensions
    """
    global _fr_debug_persons
    from models.restricted_area.detector import get_persons
    from models.face_recognition.face_detector import detect_faces_multiscale

    small_h, small_w = small_frame.shape[:2]
    sx = frame_w / small_w   # x scale: small → original
    sy = frame_h / small_h   # y scale: small → original

    # ── Step 1: YOLO person detection on fast small frame ──────────────────────
    persons_small = get_persons(small_frame)
    debug_persons = []
    results       = []

    for (px1, py1, px2, py2) in persons_small:
        # Scale person box to ORIGINAL frame coordinates
        op1 = int(px1 * sx);  oy1 = int(py1 * sy)
        op2 = int(px2 * sx);  oy2 = int(py2 * sy)
        pw  = op2 - op1;      ph  = oy2 - oy1

        # ── Person-center zone check ─────────────────────────────────────────
        if not zone_manager.is_face_inside_normalised((op1, oy1, pw, ph), frame_w, frame_h):
            continue

        debug_persons.append((op1, oy1, op2, oy2))

        # ── Step 2: crop person from ORIGINAL frame (full-res quality) ─────
        p_crop = orig_frame[oy1:oy2, op1:op2]
        if p_crop.size == 0:
            continue

        # ── Step 3: face detection inside person crop (multi-scale) ────────
        face_boxes = detect_faces_multiscale(p_crop, min_size=20)

        for face_data in face_boxes:
            if len(face_data) == 5:
                fx, fy, fw, fh, landmarks = face_data
            else:
                fx, fy, fw, fh = face_data
                landmarks = None

            # Convert face coords from person-crop space → full frame space
            abs_x = op1 + fx
            abs_y = oy1 + fy
            
            # Offset landmarks too
            if landmarks is not None:
                abs_landmarks = []
                for i in range(5):
                    abs_landmarks.append(landmarks[2*i] + op1)     # x
                    abs_landmarks.append(landmarks[2*i+1] + oy1)   # y
                face_data_abs = (abs_x, abs_y, fw, fh, abs_landmarks)
            else:
                face_data_abs = (abs_x, abs_y, fw, fh)

            # ── Face-center zone check (precise check on actual face) ───────
            face_cx = abs_x + fw // 2
            face_cy = abs_y + fh // 2
            face_in_zone = zone_manager.is_face_inside_normalised(
                (abs_x, abs_y, fw, fh), frame_w, frame_h
            )
            if not face_in_zone:
                # Face outside zone: ignore completely
                continue

            # ── Step 4: recognize on ORIGINAL frame (returns 4 values now) ───
            raw_name, p_type, conf, dist = recognize(orig_frame, face_data_abs)
            abs_landmarks_pass = face_data_abs[4] if len(face_data_abs) == 5 else None
            smooth_name, smooth_type, smooth_conf = _identity_tracker.update(
                abs_x, abs_y, fw, fh, raw_name, p_type, conf, orig_frame, landmarks=abs_landmarks_pass
            )
            results.append({
                "box":  (abs_x, abs_y, fw, fh),
                "name": smooth_name,
                "type": smooth_type,
                "conf": smooth_conf,
            })

    _identity_tracker.tick()
    _fr_debug_persons = debug_persons   # expose for draw loop
    return results


def _draw_polygon_zone(frame: np.ndarray, points: list, frame_w: int, frame_h: int):
    """Draw the polygon zone overlay on frame."""
    if not points or len(points) < 3:
        return
    pixel_pts = np.array(
        [[int(p["x"] * frame_w), int(p["y"] * frame_h)] for p in points],
        dtype=np.int32
    ).reshape((-1, 1, 2))

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pixel_pts], (255, 255, 0))
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    # Border
    cv2.polylines(frame, [pixel_pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # Label at centroid
    cx = int(np.mean([p["x"] * frame_w for p in points]))
    cy = int(np.mean([p["y"] * frame_h for p in points]))
    cv2.putText(frame, "Monitoring Zone", (cx - 60, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)


def generate_fr_frames():
    """MJPEG generator — polygon zone-aware face recognition stream."""
    global is_fr_streaming, fr_future, fr_last_results
    _fr_skip = 0          # Frame-skip counter
    _FR_PROCESS_EVERY = 4  # Process every 4th frame (stream still shows all)
    _DETECT_WIDTH = 640   # Resize to this width for detection only

    while is_fr_streaming:
        frame = camera_manager.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        frame = frame.copy()
        fh, fw = frame.shape[:2]

        zone_points = zone_manager.load_zone()

        if zone_points is None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            continue

        # Submit background recognition every Nth frame
        _fr_skip = (_fr_skip + 1) % _FR_PROCESS_EVERY
        if _fr_skip == 0 and (fr_future is None or fr_future.done()):
            if fr_future is not None:
                try:
                    fr_last_results = fr_future.result() or fr_last_results
                except Exception as e:
                    print(f"[fr_async_task] Error: {e}")
            # Downscale for fast detection
            scale = _DETECT_WIDTH / max(fw, 1)
            if scale < 1.0:
                small = cv2.resize(frame, (int(fw*scale), int(fh*scale)))
            else:
                small = frame
            fr_future = fr_executor.submit(
                _fr_async_task, small.copy(), frame.copy(), fw, fh
            )
        elif fr_future is not None and fr_future.done() and fr_last_results == []:
            try:
                fr_last_results = fr_future.result() or []
            except Exception:
                pass

        # Draw polygon zone
        _draw_polygon_zone(frame, zone_points, fw, fh)

        # Draw person boxes from last detection (debug overlay)
        for (dp1, dp2, dp3, dp4) in _fr_debug_persons:
            cv2.rectangle(frame, (dp1, dp2), (dp3, dp4), (0, 165, 255), 1)  # orange thin

        # Draw latest recognition results
        for res in fr_last_results:
            ox, oy, ow, oh = res["box"]
            smooth_name = res["name"]
            smooth_type = res["type"]

            if smooth_type == "blacklist":
                color = (0, 0, 255)
            elif smooth_type == "known":
                color = (0, 255, 0)
            else:
                color = (0, 165, 255)  # Orange for Unknown

            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), color, 2)
            
            # Format the label with similarity score
            if smooth_name != "Unknown":
                label = f"⚠ {smooth_name} ({res['conf']:.2f})" if smooth_type == "blacklist" else f"{smooth_name} ({res['conf']:.2f})"
            else:
                label = "Unknown"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (ox, oy - th - 10), (ox + tw + 8, oy), color, -1)
            cv2.putText(frame, label, (ox + 4, oy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
            # Debug: show IN ZONE tag
            cv2.putText(frame, "IN", (ox + ow - 22, oy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 150), 1)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/fr_video_feed")
def fr_video_feed():
    return Response(generate_fr_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start_fr_camera", methods=["POST"])
def start_fr_camera():
    global is_fr_streaming

    stop_all_models()

    err = _handle_source_switch(request.json)
    if err:
        return jsonify({"success": False, "message": err}), 400

    load_encodings_from_db()

    if not _open_shared_camera():
        return jsonify({"success": False, "message": "Cannot open camera."}), 500

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
        frame = camera_manager.get_latest_frame()
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
            crop             = _padded_crop(frame, x, y, w, h, pad=0.15)
            emotion, conf    = predict_emotion(crop)
            if emotion != "Unknown":
                emotion      = smooth_emotion(emotion)
            display = f"{emotion} ({int(conf*100)}%)" if emotion != "Unknown" else "Unknown"
            color   = (0, 255, 0) if emotion != "Unknown" else (140, 140, 140)
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
    stop_all_models()
    err = _handle_source_switch(request.json)
    if err:
        return jsonify({"success": False, "message": err}), 400
    if not _open_shared_camera():
        return jsonify({"success": False, "message": "Cannot open camera."}), 500
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
    global is_object_streaming, _object_frame_counter
    while is_object_streaming:
        frame = camera_manager.get_latest_frame()
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
    return Response(generate_object_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_object_camera", methods=["POST"])
def start_object_camera():
    global is_object_streaming
    stop_all_models()
    err = _handle_source_switch(request.json)
    if err:
        return jsonify({"success": False, "message": err}), 400
    if not _open_shared_camera():
        return jsonify({"success": False, "message": "Cannot open camera."}), 500
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
#  RESTRICTED AREA STREAM
# ═══════════════════════════════════════════════════════════════════════════════
def generate_restricted_frames():
    global is_restricted_streaming, restricted_frame_counter
    from models.restricted_area import process_frame

    while is_restricted_streaming:
        frame = camera_manager.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        frame = frame.copy()

        restricted_frame_counter += 1
        if restricted_frame_counter % 3 != 0:
            time.sleep(0.01)
            continue

        frame = process_frame(frame)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/restricted_video_feed")
def restricted_video_feed():
    return Response(generate_restricted_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_restricted_camera", methods=["POST"])
def start_restricted_camera():
    global is_restricted_streaming
    stop_all_models()
    err = _handle_source_switch(request.json)
    if err:
        return jsonify({"success": False, "message": err}), 400
    from models.restricted_area import load_known_persons
    load_known_persons()
    if not _open_shared_camera():
        return jsonify({"success": False, "message": "Cannot open camera."}), 500
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
            "message": f"No valid faces found in {len(files)} image(s)."
        }), 422

    return jsonify({
        "success": True,
        "message": f"Registered '{name}' with {success_count} encoding(s). ({skipped_count} skipped)"
    })


@app.route("/get_alerts")
def get_alerts():
    from models.restricted_area.database import get_recent_alerts
    return jsonify(get_recent_alerts())


# ═══════════════════════════════════════════════════════════════════════════════
#  VEHICLE DETECTION STREAM
# ═══════════════════════════════════════════════════════════════════════════════

def _delete_file_later(path, delay: int = 30):
    def _worker():
        time.sleep(delay)
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"[cleanup] Safety-deleted temp file: {path}")
        except Exception as e:
            print(f"[cleanup] Could not delete {path}: {e}")
    threading.Thread(target=_worker, daemon=True).start()


@app.route("/upload_video", methods=["POST"])
def upload_video():
    print("[upload_video] Request received")
    if 'video' not in request.files:
        return jsonify({"error": "No file attached"}), 400
    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    allowed_extensions = {'.mp4', '.avi', '.mov'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Allowed: .mp4, .avi, .mov"}), 400
    if request.content_length and request.content_length > MAX_FILE_SIZE:
        return jsonify({"error": "File too large (max 20 MB)"}), 413
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    temp_path = os.path.join(TEMP_FOLDER, file.filename)
    file.save(temp_path)
    print(f"[upload_video] Saved temporarily at: {temp_path}")
    _delete_file_later(temp_path, delay=30)
    return jsonify({"filename": file.filename})


@app.route('/video_feed/<filename>')
def plate_video_feed(filename):
    temp_path = os.path.join(TEMP_FOLDER, filename)
    return Response(
        generate_plate_frames(temp_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stop_video', methods=['POST'])
def stop_video():
    stop_video_stream()
    return jsonify({"status": "stopped"})


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Server starting on http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
