print("Starting server... loading libraries (this may take a minute)")
import time
import threading
import cv2
import numpy as np
import concurrent.futures
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

# ── AI model imports (modular structure) ──────────────────────────────────────
from models.face_recognition import (
    train_model, recognize, clear_model,
    _identity_tracker, reset_tracking_state, load_encodings_from_db
)
from models.object_detection import detect_objects, draw_detections
import os
from urllib.parse import quote as _url_quote
from dotenv import load_dotenv
load_dotenv()
from models.vehicle_detection.vehicle_model import generate_vehicle_frames, stop_video_stream

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
    global is_fr_streaming, is_object_streaming, is_restricted_streaming
    is_fr_streaming         = False
    is_object_streaming     = False
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
is_object_streaming     = False
is_restricted_streaming = False
fr_frame_counter        = 0
_object_frame_counter   = 0
restricted_frame_counter = 0

# Async recognition state
fr_executor    = concurrent.futures.ThreadPoolExecutor(max_workers=1)
fr_future      = None
fr_last_results: list  = []   # latest recognized results (name + label)
fr_detect_boxes: list  = []   # latest raw detection boxes (for immediate display)

# ── FR streaming performance ───────────────────────────────────────────────────
_DISPLAY_W       = 800         # display output width — slightly reduced for faster JPEG encode
_FR_RECOG_EVERY  = 5           # submit ArcFace every Nth detection frame
_FR_DETECT_EVERY = 2           # run SCRFD only every Nth raw frame (halves detector cost)
_fr_detect_count = 0           # raw frame counter for detection skip
_fr_recog_count  = 0           # detection-frame counter for recognition interval
_fr_recog_cache: list = []     # [{box,name,type,conf,ts}] — reuse identity for latched faces
_RECOG_CACHE_TTL = 3.0         # seconds to reuse a cached identity
_zone_cache_pts: list | None = None   # zone points cached per stream; avoids per-frame DB hit
_zone_cache_frame = 0          # frame counter for zone refresh
_ZONE_CACHE_EVERY = 60         # refresh zone every 60 raw frames (~2 s at 30 fps)


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


# ── Recognition result cache helpers (Opt 3) ──────────────────────────────────
def _box_iou(a: tuple, b: tuple) -> float:
    """IoU for two (x, y, w, h) boxes."""
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[0]+a[2], b[0]+b[2]); yB = min(a[1]+a[3], b[1]+b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / (union + 1e-5)


def _get_cached_recog(x: int, y: int, w: int, h: int) -> dict | None:
    """Return a cached recognition entry IoU-matching the box (TTL-gated), or None."""
    global _fr_recog_cache
    now = time.monotonic()
    _fr_recog_cache = [e for e in _fr_recog_cache if (now - e["ts"]) < _RECOG_CACHE_TTL]
    box = (x, y, w, h)
    for e in _fr_recog_cache:
        if _box_iou(box, e["box"]) > 0.40:
            return e
    return None


def _set_cached_recog(x: int, y: int, w: int, h: int,
                       name: str, rtype: str, conf: float):
    """Insert or refresh a recognition result in the identity cache."""
    global _fr_recog_cache
    now = time.monotonic()
    box = (x, y, w, h)
    for e in _fr_recog_cache:
        if _box_iou(box, e["box"]) > 0.40:
            e["box"] = box; e["name"] = name; e["type"] = rtype
            e["conf"] = conf; e["ts"]   = now
            return
    _fr_recog_cache.append({"box": box, "name": name, "type": rtype,
                             "conf": conf, "ts": now})


# ── Lightweight display-level IoU + EMA tracker (Opt 3) ───────────────────────
class _FRDisplayTracker:
    """Per-frame EMA tracker for detection box smoothing.

    Reduces jitter in SCRFD output without touching recognition accuracy.
    Runs synchronously every frame on the raw detect boxes before drawing.
    """
    _ALPHA    = 0.65   # weight on new detection (higher → more responsive)
    _IOU_LINK = 0.25   # min IoU to link a detection to an existing track
    _MAX_AGE  = 6      # frames a track survives without a matching detection

    def __init__(self):
        self._tracks: list = []   # [{box:(x,y,w,h), age:int}]

    def update(self, detections: list) -> list:
        for t in self._tracks:
            t["age"] += 1
        self._tracks = [t for t in self._tracks if t["age"] <= self._MAX_AGE]

        if not detections:
            return []

        used = set()
        out  = []
        a    = self._ALPHA

        for det in detections:
            dx, dy, dw, dh = det[0], det[1], det[2], det[3]
            rest = det[4:]

            best_iou, best_t = 0.0, None
            for t in self._tracks:
                if id(t) in used:
                    continue
                iou = _box_iou((dx, dy, dw, dh), t["box"])
                if iou > best_iou:
                    best_iou, best_t = iou, t

            if best_t is not None and best_iou >= self._IOU_LINK:
                px, py, pw, ph = best_t["box"]
                sx = int(a * dx + (1 - a) * px)
                sy = int(a * dy + (1 - a) * py)
                sw = int(a * dw + (1 - a) * pw)
                sh = int(a * dh + (1 - a) * ph)
                best_t["box"] = (sx, sy, sw, sh)
                best_t["age"] = 0
                used.add(id(best_t))
                out.append((sx, sy, sw, sh) + rest)
            else:
                new_t = {"box": (dx, dy, dw, dh), "age": 0}
                self._tracks.append(new_t)
                used.add(id(new_t))
                out.append(det)

        return out

    def reset(self):
        self._tracks.clear()


_fr_display_tracker = _FRDisplayTracker()


def _remap_recog_to_boxes(results: list, det_boxes: list,
                           iou_min: float = 0.25) -> list:
    """Remap stale recognition label boxes to the nearest current detected box.

    Ensures labels follow moving faces in real time between recognition updates,
    without waiting for the next ArcFace inference cycle.
    """
    if not results or not det_boxes:
        return results
    remapped = []
    for res in results:
        rx, ry, rw, rh = res["box"]
        best_iou, best_box = 0.0, None
        for det in det_boxes:
            iou = _box_iou((rx, ry, rw, rh), (det[0], det[1], det[2], det[3]))
            if iou > best_iou:
                best_iou, best_box = iou, (det[0], det[1], det[2], det[3])
        if best_box is not None and best_iou >= iou_min:
            remapped.append({**res, "box": best_box})
        else:
            remapped.append(res)
    return remapped


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

@app.route("/object-detection")
def object_detection_page():
    return render_template("object_detection.html")

@app.route("/vehicle-detection")
def vehicle_detection_page():
    return render_template("vehicle_detection.html")

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
    from models.face_recognition.fr_database import get_unknown_dashboard, get_unknown_by_date
    search      = request.args.get("search", "").strip()
    sort_by     = request.args.get("sort", "last_seen")
    date_filter = request.args.get("date", "").strip()
    page        = int(request.args.get("page", 1))
    per_page    = int(request.args.get("per_page", 50))
    if date_filter:
        docs, total = get_unknown_by_date(date_filter, page, per_page)
    else:
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


# ── Attendance (day-wise) ──────────────────────────────────────────────────────
@app.route("/api/report/attendance")
def api_report_attendance():
    from models.face_recognition.fr_database import get_attendance_by_date, to_ist
    import datetime as _dt
    date_str = request.args.get("date", "").strip()
    search   = request.args.get("search", "").strip()
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    if not date_str:
        date_str = to_ist(_dt.datetime.utcnow()).strftime("%Y-%m-%d")
    docs, total = get_attendance_by_date(date_str, search, page, per_page)
    return jsonify({"data": docs, "total": total, "page": page, "per_page": per_page, "date": date_str})


@app.route("/api/report/attendance/dates")
def api_attendance_dates():
    from models.face_recognition.fr_database import get_attendance_dates
    return jsonify({"dates": get_attendance_dates()})


@app.route("/api/report/unknown/dates")
def api_unknown_dates():
    from models.face_recognition.fr_database import get_unknown_dates
    return jsonify({"dates": get_unknown_dates()})


@app.route("/api/report/blacklist/dates")
def api_blacklist_dates():
    from models.face_recognition.fr_database import get_blacklist_dates
    return jsonify({"dates": get_blacklist_dates()})


# ── Daily Summary ──────────────────────────────────────────────────────────────
@app.route("/api/report/summary")
def api_report_summary():
    from models.face_recognition.fr_database import get_daily_summary, to_ist
    import datetime as _dt
    date_str = request.args.get("date", "").strip()
    if not date_str:
        date_str = to_ist(_dt.datetime.utcnow()).strftime("%Y-%m-%d")
    return jsonify(get_daily_summary(date_str))


@app.route("/api/report/summary/dates")
def api_summary_dates():
    from models.face_recognition.fr_database import get_summary_dates
    return jsonify({"dates": get_summary_dates()})


# ── Excel Export ───────────────────────────────────────────────────────────────
@app.route("/api/report/export/xlsx")
def api_export_xlsx():
    from models.face_recognition.fr_database import export_attendance_xlsx, to_ist
    from flask import make_response
    import datetime as _dt
    date_str = request.args.get("date", "").strip()
    if not date_str:
        date_str = to_ist(_dt.datetime.utcnow()).strftime("%Y-%m-%d")
    data = export_attendance_xlsx(date_str)
    if not data:
        return jsonify({"error": "Failed to generate Excel file"}), 500
    resp = make_response(data)
    resp.headers["Content-Type"] = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    resp.headers["Content-Disposition"] = f"attachment; filename=attendance_{date_str}.xlsx"
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

def _fr_detect_zone_roi(orig_frame: np.ndarray, zone_snapshot: list,
                        frame_w: int, frame_h: int) -> list:
    """
    SYNC detection step — called every _FR_DETECT_EVERY raw frame for immediate box display.

    Strategy:
        1. Detect on full frame (not just ROI crop) to avoid coord-mapping bugs.
        2. Apply zone polygon filter AFTER detection + NMS (not before) to prevent
           zone boundary from generating duplicate detections.
        3. Run 1.5× upscale pass only when no/tiny faces found in pass 1.

    Returns list of (x, y, w, h, landmarks_or_None, conf) in original-frame coords.
    """
    from models.face_recognition.face_detector import get_faces_dnn
    from zone_manager import zone_manager as _zm

    if not zone_snapshot or len(zone_snapshot) < 3:
        return []

    pixel_pts = [
        (int(p["x"] * frame_w), int(p["y"] * frame_h))
        for p in zone_snapshot
    ]

    # Detect on full frame — avoid ROI crop creating coordinate mapping drift
    all_faces = get_faces_dnn(orig_frame, smooth=False, min_size=18)

    # Apply upscale pass for far faces when needed
    if not all_faces or all(f[2] < 40 for f in all_faces):
        from models.face_recognition.face_detector import _scrfd_infer, _raw_to_xywh, _nms, _NMS_IOU
        up       = 1.5
        nw2      = min(int(frame_w * up), 1920)
        nh2      = min(int(frame_h * up), 1920)
        up_frame = cv2.resize(orig_frame, (nw2, nh2), interpolation=cv2.INTER_LINEAR)
        inv      = 1.0 / up
        raw_up   = []
        for (x1, y1, x2, y2, conf, kps) in _scrfd_infer(up_frame):
            raw_up.append((x1 * inv, y1 * inv, x2 * inv, y2 * inv, conf, [v * inv for v in kps]))
        if raw_up:
            up_faces, up_boxes, up_scores = _raw_to_xywh(raw_up, frame_w, frame_h, 18, orig_frame)
            all_raw_boxes = [(f[0], f[1], f[2], f[3]) for f in all_faces] + up_boxes
            all_raw_scores = [f[5] for f in all_faces] + up_scores
            kept_idx = _nms(all_raw_boxes, all_raw_scores, _NMS_IOU)
            merged = (all_faces + up_faces)
            all_faces = [merged[i] for i in kept_idx]

    # Filter: only keep faces whose centre is inside the zone polygon
    in_zone = [f for f in all_faces if _zm.is_face_inside_zone((f[0], f[1], f[2], f[3]), pixel_pts)]
    return in_zone


def _fr_recog_task(orig_frame: np.ndarray, face_boxes: list,
                   frame_w: int, frame_h: int, zone_snapshot: list) -> list:
    """
    ASYNC recognition step.
    Takes pre-detected + zone-filtered face boxes from _fr_detect_zone_roi.
    Deduplicates overlapping boxes via IoU before ArcFace to prevent same face
    being recognised twice (root cause of duplicate 'Unknown' labels).

    Returns list of {box, name, type, conf} dicts.
    """
    results = []

    if not zone_snapshot or len(zone_snapshot) < 3 or not face_boxes:
        return results

    fh_f, fw_f = orig_frame.shape[:2]

    # ── Step A: IoU-based dedup on face_boxes BEFORE recognition ─────────────
    # Prevents the same physical face from creating two tracker/recognition slots.
    deduped_boxes: list = []
    for fd in face_boxes:
        if len(fd) < 5:
            continue
        x, y, w, h = fd[0], fd[1], fd[2], fd[3]
        dup = False
        for ex in deduped_boxes:
            if _box_iou((x, y, w, h), (ex[0], ex[1], ex[2], ex[3])) > 0.50:
                # Keep the higher-confidence one
                if (float(fd[5]) if len(fd) >= 6 else 0.5) > (float(ex[5]) if len(ex) >= 6 else 0.5):
                    deduped_boxes.remove(ex)
                    deduped_boxes.append(fd)
                dup = True
                break
        if not dup:
            deduped_boxes.append(fd)

    for face_data in deduped_boxes:
        x, y, w, h  = face_data[0], face_data[1], face_data[2], face_data[3]
        landmarks    = face_data[4]
        det_conf     = float(face_data[5]) if len(face_data) >= 6 else 0.5

        # Gate 1: confidence — raised from 0.20 to 0.30 to reduce FP
        if det_conf < 0.30:
            continue

        # Gate 2: minimum face size
        if w < 18 or h < 18:
            continue

        # Gate 3: valid landmarks (strongest FP filter — walls/plants fail this)
        if landmarks is None:
            continue

        # Gate 4: aspect ratio — tighter range to kill profile/tilted FPs
        aspect = w / (h + 1e-5)
        if aspect < 0.45 or aspect > 1.9:
            continue

        # Gate 5: face sharpness on raw crop
        x1c = max(0, x);     y1c = max(0, y)
        x2c = min(fw_f, x+w); y2c = min(fh_f, y+h)
        face_crop = orig_frame[y1c:y2c, x1c:x2c]
        if face_crop.size == 0:
            continue
        blur_score = float(cv2.Laplacian(
            cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if blur_score < 8.0:
            continue

        # Identity cache — skip ArcFace for faces seen within _RECOG_CACHE_TTL
        cached = _get_cached_recog(x, y, w, h)
        if cached is not None:
            smooth_name, smooth_type, smooth_conf = _identity_tracker.update(
                x, y, w, h, cached["name"], cached["type"], cached["conf"], orig_frame,
                landmarks=landmarks, embedding=None,
                camera_source=camera_source_config.get("type", "webcam"),
                zone_id="Monitoring Zone",
            )
            _set_cached_recog(x, y, w, h, smooth_name, smooth_type, smooth_conf)
            results.append({"box": (x, y, w, h), "name": smooth_name,
                            "type": smooth_type, "conf": smooth_conf})
            continue

        # Full ArcFace recognition (slow path)
        raw_name, p_type, conf, dist, emb = recognize(orig_frame, face_data)

        smooth_name, smooth_type, smooth_conf = _identity_tracker.update(
            x, y, w, h, raw_name, p_type, conf, orig_frame,
            landmarks=landmarks, embedding=emb,
            camera_source=camera_source_config.get("type", "webcam"),
            zone_id="Monitoring Zone",
        )

        _set_cached_recog(x, y, w, h, smooth_name, smooth_type, smooth_conf)

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


def generate_fr_frames():
    """
    MJPEG generator — optimised two-stage FR pipeline.

    Stage 1 (sync, every _FR_DETECT_EVERY raw frames) — SCRFD → raw boxes → draw immediately.
    Stage 2 (async, every _FR_RECOG_EVERY detection frames) — ArcFace → update labels.
    Between detection frames the previous smoothed boxes are used for display (free).

    Optimisations:
      Opt 1 — Display resize     : MJPEG output at _DISPLAY_W wide.
      Opt 2 — Detection skip     : SCRFD runs every 2nd raw frame.
      Opt 3 — Recog interval     : ArcFace submitted every 5th detection frame.
      Opt 4 — Identity cache     : latched faces skip ArcFace (2–3 s TTL).
      Opt 5 — Zone cache         : zone loaded from DB only every 60 frames.
      Opt 6 — Cond. upscale      : zone ROI upscaled only when no/tiny faces.
      Opt 7 — JPEG 72 quality    : smaller MJPEG payload.
    """
    global is_fr_streaming, fr_future, fr_last_results, fr_detect_boxes
    global _fr_detect_count, _fr_recog_count
    global _zone_cache_pts, _zone_cache_frame

    raw_frame_count = 0

    while is_fr_streaming:
        frame = camera_manager.get_latest_frame()
        if frame is None:
            time.sleep(0.005)
            continue
        fh, fw = frame.shape[:2]
        raw_frame_count += 1

        # Opt 1: Scale display frame — detection still on full-res original
        if fw > _DISPLAY_W:
            ds     = _DISPLAY_W / fw
            disp_h = int(fh * ds)
            disp   = cv2.resize(frame, (_DISPLAY_W, disp_h), interpolation=cv2.INTER_LINEAR)
        else:
            ds   = 1.0
            disp = frame.copy()
        d_h, d_w = disp.shape[:2]

        # Opt 5: Refresh zone from DB only every _ZONE_CACHE_EVERY frames
        _zone_cache_frame += 1
        if _zone_cache_pts is None or _zone_cache_frame >= _ZONE_CACHE_EVERY:
            _zone_cache_pts   = zone_manager.load_zone()
            _zone_cache_frame = 0
        zone_points = _zone_cache_pts

        if zone_points is None:
            fr_last_results = []
            fr_detect_boxes = []
        else:
            # Opt 2: Run SCRFD detection only every _FR_DETECT_EVERY raw frames
            _fr_detect_count += 1
            if _fr_detect_count >= _FR_DETECT_EVERY:
                _fr_detect_count = 0
                raw_boxes       = _fr_detect_zone_roi(frame, list(zone_points), fw, fh)
                fr_detect_boxes = _fr_display_tracker.update(raw_boxes)

            # Stage 2: Async ArcFace — collect done result; submit every Nth detection frame
            if fr_future is not None and fr_future.done():
                try:
                    res = fr_future.result()
                    if res is not None:
                        fr_last_results = res
                except Exception as e:
                    print(f"[fr_recog_task] Error: {e}")
                fr_future = None

            _fr_recog_count += 1
            if (fr_detect_boxes and fr_future is None
                    and _fr_recog_count >= _FR_RECOG_EVERY):
                _fr_recog_count = 0
                fr_future = fr_executor.submit(
                    _fr_recog_task,
                    frame.copy(), list(fr_detect_boxes), fw, fh, list(zone_points)
                )

        # Opt 4 (label remap): pin stale recognition labels to current tracked boxes
        display_results = _remap_recog_to_boxes(fr_last_results, fr_detect_boxes)

        # Draw zone overlay on display frame
        _draw_polygon_zone(disp, zone_points, d_w, d_h)

        # Draw Stage-1 detection boxes (thin blue = "detecting…") on display frame.
        for det in fr_detect_boxes:
            det_box = (det[0], det[1], det[2], det[3])
            covered = any(
                _box_iou(det_box, (r["box"][0], r["box"][1], r["box"][2], r["box"][3])) > 0.30
                for r in display_results
            )
            if not covered:
                bx = int(det[0] * ds); by = int(det[1] * ds)
                bw = max(1, int(det[2] * ds)); bh = max(1, int(det[3] * ds))
                cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (120, 120, 220), 1)

        # Draw Stage-2 recognition results (colored + labeled) on display frame
        for res in display_results:
            ox = int(res["box"][0] * ds); oy = int(res["box"][1] * ds)
            ow = max(1, int(res["box"][2] * ds)); oh = max(1, int(res["box"][3] * ds))
            smooth_name = res["name"]
            smooth_type = res["type"]

            if smooth_type == "blacklist":
                color = (0, 0, 255)
            elif smooth_type == "known":
                color = (0, 255, 0)
            else:
                color = (0, 165, 255)

            cv2.rectangle(disp, (ox, oy), (ox + ow, oy + oh), color, 2)

            if smooth_name != "Unknown":
                label = (f"! {smooth_name} ({res['conf']:.2f})"
                         if smooth_type == "blacklist"
                         else f"{smooth_name} ({res['conf']:.2f})")
            else:
                label = "Unknown"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(disp, (ox, oy - th - 10), (ox + tw + 8, oy), color, -1)
            cv2.putText(disp, label, (ox + 4, oy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        # Opt 7: JPEG 72 for stream — smaller frames, faster browser decode
        _, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 72])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.route("/fr_video_feed")
def fr_video_feed():
    return Response(generate_fr_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start_fr_camera", methods=["POST"])
def start_fr_camera():
    global is_fr_streaming, _fr_detect_count, _fr_recog_count, _fr_recog_cache
    global _zone_cache_pts, _zone_cache_frame

    stop_all_models()

    err = _handle_source_switch(request.json)
    if err:
        return jsonify({"success": False, "message": err}), 400

    load_encodings_from_db()
    zone_manager.reload()   # re-read zone from DB (picks up zones saved while camera was off)

    if not _open_shared_camera():
        return jsonify({"success": False, "message": "Cannot open camera."}), 500

    _fr_detect_count = 0
    _fr_recog_count  = 0
    _fr_recog_cache  = []
    _zone_cache_pts  = None   # force fresh zone load on first frame
    _zone_cache_frame = 0
    _fr_display_tracker.reset()
    is_fr_streaming  = True
    return jsonify({"success": True})


@app.route("/stop_fr_camera", methods=["POST"])
def stop_fr_camera():
    global is_fr_streaming, _fr_recog_cache
    is_fr_streaming = False
    _fr_recog_cache = []
    _fr_display_tracker.reset()
    if not is_object_streaming and not is_restricted_streaming:
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
    if not is_fr_streaming and not is_restricted_streaming:
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


@app.route('/vehicle_video_feed/<filename>')
def vehicle_video_feed(filename):
    temp_path = os.path.join(TEMP_FOLDER, filename)
    return Response(
        generate_vehicle_frames(temp_path),
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
