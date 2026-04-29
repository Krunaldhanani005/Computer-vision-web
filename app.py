print("Starting server... loading libraries (this may take a minute)")
import time
import threading
import cv2
import numpy as np
import scipy.optimize
import concurrent.futures
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

# ── AI model imports (modular structure) ──────────────────────────────────────
from models.face_recognition import (
    train_model, recognize, clear_model,
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
from ra_camera_manager import ra_camera_manager
from zone_manager import zone_manager

camera_source_config = {"type": "webcam", "url": None}

# Hardcoded RA CCTV stream (@ in password URL-encoded as %40)
_RA_CCTV_URL = "rtsp://Test:Nanta%40123@192.168.29.118:554/1/1?transmode=unicast&profile=vam"


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
        # Accept direct URL from request (rtsp://, http://, https://)
        direct_url = (data.get("url") or "").strip()
        if direct_url:
            url = direct_url
        else:
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

ra_executor    = concurrent.futures.ThreadPoolExecutor(max_workers=1)
ra_future      = None
ra_last_results: list = []


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
    from models.face_recognition.fr_database import get_known_dashboard
    search   = request.args.get("search", "").strip()
    sort_by  = request.args.get("sort", "last_seen")
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 24))
    docs, total = get_known_dashboard(search, sort_by, page, per_page)
    return jsonify({"data": docs, "total": total, "page": page, "per_page": per_page})


@app.route("/api/report/known/stats")
def api_report_known_stats():
    from models.face_recognition.fr_database import get_known_stats
    return jsonify(get_known_stats())


@app.route("/api/report/unknown")
def api_report_unknown():
    from models.face_recognition.fr_database import get_unknown_dashboard
    search   = request.args.get("search", "").strip()
    sort_by  = request.args.get("sort", "last_seen")
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    docs, total = get_unknown_dashboard(search, sort_by, page, per_page)
    return jsonify({"data": docs, "total": total, "page": page, "per_page": per_page})

@app.route("/api/report/unknown/delete_all", methods=["POST"])
def api_report_unknown_delete_all():
    from models.face_recognition.fr_database import delete_all_unknown_logs
    success = delete_all_unknown_logs()
    if success:
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Failed to delete reports"}), 500



@app.route("/api/report/unknown/stats")
def api_report_unknown_stats():
    from models.face_recognition.fr_database import get_unknown_stats
    return jsonify(get_unknown_stats())


@app.route("/api/report/blacklist")
def api_report_blacklist():
    from models.face_recognition.fr_database import get_blacklist_dashboard
    search      = request.args.get("search", "").strip()
    date_filter = request.args.get("date", "").strip()
    page        = int(request.args.get("page", 1))
    per_page    = int(request.args.get("per_page", 50))
    docs, total = get_blacklist_dashboard(search, date_filter, page, per_page)
    return jsonify({"data": docs, "total": total, "page": page, "per_page": per_page})


@app.route("/api/report/blacklist/stats")
def api_report_blacklist_stats():
    from models.face_recognition.fr_database import get_blacklist_stats
    return jsonify(get_blacklist_stats())


@app.route("/api/report/export/known")
def api_export_known():
    from models.face_recognition.fr_database import export_known_csv
    from flask import make_response
    csv_data = export_known_csv()
    resp = make_response(csv_data)
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=known_persons_report.csv"
    return resp


@app.route("/api/report/export/blacklist")
def api_export_blacklist():
    from models.face_recognition.fr_database import export_blacklist_csv
    from flask import make_response
    csv_data = export_blacklist_csv()
    resp = make_response(csv_data)
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=blacklist_alerts_report.csv"
    return resp

@app.route("/ra-report")
def ra_report_page():
    return render_template("ra_report.html")

@app.route("/api/ra/stats")
def api_ra_stats():
    from models.restricted_area.database import get_ra_stats
    return jsonify(get_ra_stats())

@app.route("/api/report/restricted_area")
def api_report_restricted_area():
    from models.restricted_area.database import get_restricted_dashboard
    search      = request.args.get("search", "").strip()
    date_filter = request.args.get("date", "").strip()
    page        = int(request.args.get("page", 1))
    per_page    = int(request.args.get("per_page", 50))
    docs, total = get_restricted_dashboard(search, date_filter, page, per_page)
    return jsonify({"data": docs, "total": total, "page": page, "per_page": per_page})

@app.route("/api/report/export/restricted_area")
def api_export_restricted_area():
    from models.restricted_area.database import export_restricted_csv
    from flask import make_response
    csv_data = export_restricted_csv()
    resp = make_response(csv_data)
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=restricted_area_report.csv"
    return resp

# ── RA Zone Routes (separate from FR zones) ──────────────────────────────────
@app.route("/ra/save_zone", methods=["POST"])
def ra_save_zone():
    from models.restricted_area.database import save_ra_zone, clear_ra_zone
    global ra_last_results
    data = request.json
    zone_data = data.get("zone", None)
    
    # Clear active detections on zone change
    ra_last_results = []
    
    if zone_data is None:
        clear_ra_zone("restricted_default")
        return jsonify({"success": True})
    if isinstance(zone_data, list):
        ok = save_ra_zone(zone_data, "restricted_default")
        print(f"[ra_zone] Saved {len(zone_data)} pts → {ok}")
        return jsonify({"success": ok})
    return jsonify({"success": False, "error": "Invalid zone format"}), 400

@app.route("/ra/get_zone", methods=["GET"])
def ra_get_zone():
    from models.restricted_area.database import load_ra_zone
    pts = load_ra_zone("restricted_default")
    return jsonify({"zone": pts})

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
    from models.face_recognition.fr_database import get_known_dashboard
    docs, total = get_known_dashboard()
    return jsonify({"data": docs, "total": total})


@app.route("/api/fr/unknown")
def api_unknown():
    from models.face_recognition.fr_database import get_unknown_dashboard
    docs, total = get_unknown_dashboard()
    return jsonify({"data": docs, "total": total})


@app.route("/api/fr/alerts")
def api_alerts():
    from models.face_recognition.fr_database import get_blacklist_dashboard
    docs, total = get_blacklist_dashboard()
    return jsonify({"data": docs, "total": total})


# ═══════════════════════════════════════════════════════════════════════════════
#  FACE RECOGNITION STREAM  (polygon zone-aware)
# ═══════════════════════════════════════════════════════════════════════════════

def _fr_async_task(orig_frame: np.ndarray, frame_w: int, frame_h: int,
                   zone_snapshot: list):
    """
    Background recognition task.

    zone_snapshot: copy of active zone points captured at submission time.
        Using a snapshot (not re-reading zone_manager inside the task) prevents
        a race where the zone is cleared while the task is in-flight.

    Quality gates (all must pass):
      1. Detection confidence >= 0.42
      2. Face size >= 50x50 px
      3. Valid 5-point landmarks
      4. Face aspect ratio 0.4–2.0
      5. Sharpness (Laplacian) >= 12
      6. Skin-tone ratio >= 15%  (eliminates glass/furniture CCTV false positives)
      7. (zone snapshot guard)
      8. Face centre inside zone polygon
    """
    from models.face_recognition.face_detector import detect_faces_multiscale

    results = []

    # No zone at snapshot time → task should not have been submitted, but guard anyway
    if zone_snapshot is None or len(zone_snapshot) < 3:
        return results

    face_boxes = detect_faces_multiscale(orig_frame, min_size=45)

    for face_data in face_boxes:
        if len(face_data) < 5:
            continue

        x, y, w, h = face_data[0], face_data[1], face_data[2], face_data[3]
        landmarks  = face_data[4]
        det_conf   = float(face_data[5]) if len(face_data) >= 6 else 0.5

        # Gate 1: detection confidence — 0.25 for CCTV (wider recall, quality gates downstream filter FPs)
        if det_conf < 0.25:
            continue

        # Gate 2: minimum face size (45px — aligned with _MIN_SAVE_W/H)
        if w < 45 or h < 45:
            continue

        # Gate 3: valid landmarks (strongest false-positive filter)
        if landmarks is None:
            continue

        # Gate 4: aspect ratio
        aspect = w / (h + 1e-5)
        if aspect < 0.4 or aspect > 2.0:
            continue

        # Gate 5: sharpness + crop extraction (used by Gate 6 too)
        fh_f, fw_f = orig_frame.shape[:2]
        x1c, y1c   = max(0, x), max(0, y)
        x2c, y2c   = min(fw_f, x + w), min(fh_f, y + h)
        face_crop  = orig_frame[y1c:y2c, x1c:x2c]
        if face_crop.size == 0:
            continue
        gray_crop  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
        if blur_score < 15.0:   # Fix #4: aligned with _MIN_SHARPNESS
            continue

        # Gate 6: skin-tone filter — eliminates glass, furniture, signage false positives.
        # YCrCb range covers dark to fair skin under normal and IR-assisted colour cameras.
        # Real faces need >= 15% skin-coloured pixels; glass/wood/walls score near 0%.
        ycrcb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        skin_mask  = cv2.inRange(
            ycrcb_crop,
            np.array([0,   133,  77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8),
        )
        skin_ratio = float(np.count_nonzero(skin_mask)) / max(skin_mask.size, 1)
        if skin_ratio < 0.15:
            continue

        # Gate 8: face centre inside zone (using snapshot — no live zone_manager read)
        from zone_manager import zone_manager as _zm
        if not _zm.is_face_inside_zone((x, y, w, h),
                                       [(int(p["x"] * frame_w), int(p["y"] * frame_h))
                                        for p in zone_snapshot]):
            continue

        # Gate 7 (Fix #5): Frontal face check — both eyes present + eye dist >= 20px
        from models.face_recognition.face_recognition_model import _is_frontal_face
        if not _is_frontal_face(landmarks):
            continue

        # Recognize — returns (name, type, conf, dist, embedding)
        raw_name, p_type, conf, dist, emb = recognize(orig_frame, face_data)

        zone_id = "Monitoring Zone"
        smooth_name, smooth_type, smooth_conf = _identity_tracker.update(
            x, y, w, h, raw_name, p_type, conf, orig_frame,
            landmarks=landmarks, embedding=emb,
            camera_source=camera_source_config.get("type", "webcam"),
            zone_id=zone_id,
        )

        results.append({
            "box":  (x, y, w, h),
            "name": smooth_name,
            "type": smooth_type,
            "conf": smooth_conf,
        })

    _identity_tracker.tick()
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


def _iou(bA, bB):
    xA = max(bA[0], bB[0])
    yA = max(bA[1], bB[1])
    xB = min(bA[0] + bA[2], bB[0] + bB[2])
    yB = min(bA[1] + bA[3], bB[1] + bB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = bA[2]*bA[3] + bB[2]*bB[3] - inter
    return inter / (union + 1e-5)

class ByteTracker:
    """ByteTrack-style IoU tracker with velocity extrapolation between skipped frames."""
    def __init__(self):
        self.trackers = []
    
    def init_from_results(self, frame, results):
        if not results:
            self.trackers = []
            return
            
        new_trackers = []
        if not self.trackers:
            for res in results:
                t = res.copy()
                t['vx'] = 0.0
                t['vy'] = 0.0
                new_trackers.append(t)
            self.trackers = new_trackers
            return
            
        cost_matrix = np.ones((len(self.trackers), len(results)))
        for i, t in enumerate(self.trackers):
            for j, r in enumerate(results):
                cost_matrix[i, j] = 1.0 - _iou(t['box'], r['box'])
                
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        
        matched_results = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.8: # IoU > 0.2
                t = self.trackers[i]
                r = results[j]
                
                old_x, old_y, w, h = t['box']
                new_x, new_y, nw, nh = r['box']
                
                vx = new_x - old_x
                vy = new_y - old_y
                
                new_t = r.copy()
                new_t['vx'] = 0.5 * t['vx'] + 0.5 * vx
                new_t['vy'] = 0.5 * t['vy'] + 0.5 * vy
                new_trackers.append(new_t)
                matched_results.add(j)
                
        for j, r in enumerate(results):
            if j not in matched_results:
                t = r.copy()
                t['vx'] = 0.0
                t['vy'] = 0.0
                new_trackers.append(t)
                
        self.trackers = new_trackers

    def update(self, frame):
        if not self.trackers: return []
        results = []
        fh, fw = frame.shape[:2]
        for t in self.trackers:
            x, y, w, h = t['box']
            nx = int(x + t['vx'])
            ny = int(y + t['vy'])
            nx = max(0, min(fw - w, nx))
            ny = max(0, min(fh - h, ny))
            t['box'] = (nx, ny, w, h)
            results.append({"box": t['box'], "name": t['name'], "type": t['type'], "conf": t['conf']})
        return results

_fr_fast_tracker = ByteTracker()

def generate_fr_frames():
    """MJPEG generator — polygon zone-aware face recognition stream."""
    global is_fr_streaming, fr_future, fr_last_results
    _fr_skip = 0
    # Fix #2: Process every 2nd frame — better walking-face continuity than every 5th
    FR_FRAME_SKIP     = 2
    _FR_PROCESS_EVERY = FR_FRAME_SKIP

    while is_fr_streaming:
        frame = camera_manager.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        frame = frame.copy()
        fh, fw = frame.shape[:2]

        zone_points = zone_manager.load_zone()

        if zone_points is None:
            # NO ZONE = NO DETECTION
            fr_last_results = []
            _fr_fast_tracker.trackers = []
            _fr_skip = 0
        else:
            _fr_skip = (_fr_skip + 1) % _FR_PROCESS_EVERY
            if _fr_skip == 0 and (fr_future is None or fr_future.done()):
                if fr_future is not None:
                    try:
                        res = fr_future.result()
                        if res is not None:
                            _fr_fast_tracker.init_from_results(frame, res)
                    except Exception as e:
                        print(f"[fr_async_task] Error: {e}")

                # Pass zone snapshot so the task doesn't need to re-read zone_manager
                fr_future = fr_executor.submit(
                    _fr_async_task, frame.copy(), fw, fh, list(zone_points)
                )
            elif fr_future is not None and fr_future.done() and not _fr_fast_tracker.trackers:
                try:
                    res = fr_future.result()
                    if res is not None:
                        _fr_fast_tracker.init_from_results(frame, res)
                except Exception:
                    pass

            fr_last_results = _fr_fast_tracker.update(frame)

        # Draw polygon zone
        _draw_polygon_zone(frame, zone_points, fw, fh)

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
    zone_manager.reload()   # re-read zone from DB (picks up zones saved while camera was off)

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
#  RESTRICTED AREA STREAM  (uses dedicated ra_camera_manager — separate from FR)
# ═══════════════════════════════════════════════════════════════════════════════

# RA-specific source config (independent from FR's camera_source_config)
_ra_source_config = {"type": "webcam", "url": None}

# Predefined CCTV stream for RA (@ in password kept raw — ffmpeg parses last @ as host sep)
_RA_CCTV_URL = "rtsp://Test:Nanta@123@192.168.29.118:554/1/1?transmode=unicast&profile=vam"


def generate_restricted_frames():
    global is_restricted_streaming, ra_future, ra_last_results
    from models.restricted_area import process_frame
    from models.restricted_area.database import load_ra_zone

    _ra_skip = 0
    _RA_PROCESS_EVERY = 2

    # Cache zone locally — reload every 5 s instead of every frame
    _ra_zone_cache     = load_ra_zone("restricted_default")
    _ra_zone_reload_ts = time.monotonic()
    _RA_ZONE_REFRESH   = 5.0

    while is_restricted_streaming:
        frame = ra_camera_manager.get_latest_frame()   # ← dedicated RA camera
        if frame is None:
            time.sleep(0.01)
            continue
        frame = frame.copy()
        fh, fw = frame.shape[:2]

        # Refresh zone cache every 5 s (picks up save/delete without per-frame DB hit)
        _now = time.monotonic()
        if (_now - _ra_zone_reload_ts) >= _RA_ZONE_REFRESH:
            _ra_zone_cache     = load_ra_zone("restricted_default", force=True)
            _ra_zone_reload_ts = _now

        ra_z_pts = _ra_zone_cache

        # ── Zone Required Gate ──────────────────────────────────────────────
        if ra_z_pts is None or len(ra_z_pts) < 3:
            # NO zone = NO detection. Clear results.
            ra_last_results = []
            if ra_future is not None and not ra_future.done():
                ra_future.cancel()
        else:
            _ra_skip = (_ra_skip + 1) % _RA_PROCESS_EVERY
            if _ra_skip == 0 and (ra_future is None or ra_future.done()):
                if ra_future is not None:
                    try:
                        ra_last_results = ra_future.result() or ra_last_results
                    except Exception:
                        pass

                c_source = _ra_source_config.get("type", "webcam")
                z_id     = "Restricted Zone"

                ra_future = ra_executor.submit(
                    process_frame, frame.copy(), c_source, z_id
                )
            elif ra_future is not None and ra_future.done() and not ra_last_results:
                try:
                    ra_last_results = ra_future.result() or []
                except Exception:
                    pass

        # ── Draw RA Zone Overlay (same as FR) ───────────────────────────────
        if ra_z_pts and len(ra_z_pts) >= 3:
            pixel_pts = np.array(
                [[int(p["x"] * fw), int(p["y"] * fh)] for p in ra_z_pts],
                dtype=np.int32
            ).reshape((-1, 1, 2))
            
            # Semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pixel_pts], (0, 120, 255))  # Orange for RA
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
            
            # Border
            cv2.polylines(frame, [pixel_pts], isClosed=True, color=(0, 120, 255), thickness=2)
            
            # Label
            cx = int(np.mean([p["x"] * fw for p in ra_z_pts]))
            cy = int(np.mean([p["y"] * fh for p in ra_z_pts]))
            cv2.putText(frame, "Restricted Zone", (cx - 60, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)

        # ── Draw detection results on frame ─────────────────────────────────
        for res in ra_last_results:
            x1, y1, x2, y2 = res["box"]
            color = res["color"]
            label = res["label"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/restricted_video_feed")
def restricted_video_feed():
    return Response(generate_restricted_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start_restricted_camera", methods=["POST"])
def start_restricted_camera():
    global is_restricted_streaming, ra_last_results

    # Stop RA only — never touch FR or shared camera
    is_restricted_streaming = False
    ra_last_results = []
    ra_camera_manager.close_camera()   # release RA camera cleanly
    time.sleep(0.3)                    # brief settle before re-open

    data   = request.json or {}
    source = data.get("source", "webcam")

    if source == "cctv":
        _ra_source_config["type"] = "cctv"
        _ra_source_config["url"]  = _RA_CCTV_URL
        ok = ra_camera_manager.open_cctv(_RA_CCTV_URL)
        print(f"[RA] open_cctv → {ok}  url={_RA_CCTV_URL}")
    else:
        _ra_source_config["type"] = "webcam"
        _ra_source_config["url"]  = None
        ok = ra_camera_manager.open_webcam(0)
        print(f"[RA] open_webcam(0) → {ok}")

    if not ok:
        return jsonify({"success": False, "message": "Cannot open camera. Check device or RTSP stream."}), 500

    from models.restricted_area import load_known_persons
    from models.restricted_area.database import load_ra_zone
    load_known_persons()
    load_ra_zone("restricted_default", force=True)  # Reload zone from DB on start
    is_restricted_streaming = True
    return jsonify({"success": True})


@app.route("/stop_restricted_camera", methods=["POST"])
def stop_restricted_camera():
    global is_restricted_streaming, ra_last_results
    is_restricted_streaming = False
    ra_last_results = []
    ra_camera_manager.close_camera()   # only RA camera — FR camera untouched
    return jsonify({"success": True})


@app.route("/add_known_person", methods=["POST"])
def add_known_person():
    from models.restricted_area.face_handler import extract_encoding_from_image
    from models.restricted_area.database    import insert_ra_known_person

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
        if insert_ra_known_person(name, encoding.tolist() if hasattr(encoding, 'tolist') else encoding):
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
    from models.restricted_area.database import get_ra_dashboard
    docs, _ = get_ra_dashboard("", "", 1, 20)
    return jsonify(docs)


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
