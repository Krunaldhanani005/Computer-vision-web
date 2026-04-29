import cv2
import time
import os
from collections import defaultdict, deque
from ultralytics import YOLO
from .ocr_utils import extract_plate_text

# Lightweight model for smooth CPU streaming
vehicle_model = YOLO("yolov8n.pt")

# Dedicated plate model if available
try:
    plate_model = YOLO("best_plate_model.pt") if os.path.exists("best_plate_model.pt") else None
except Exception:
    plate_model = None

# Haar cascade fallback
plate_cascade = None
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_russian_plate_number.xml')
if os.path.exists(cascade_path):
    plate_cascade = cv2.CascadeClassifier(cascade_path)

stop_vehicle_video = False

# ─────────────────────────────────────────────────────────────────────────────
# STABILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Per-object histories (keyed by integer obj_id)
_class_history = defaultdict(lambda: deque(maxlen=5))
_box_history   = defaultdict(lambda: deque(maxlen=5))


def _compute_iou(boxA, boxB):
    """Compute Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _match_to_track(new_box, tracked_boxes, iou_thresh=0.35):
    """
    Return the track ID of the existing box with highest IoU (if above threshold),
    otherwise return a new unique ID.
    """
    best_id, best_iou = None, iou_thresh
    for tid, tbox in tracked_boxes.items():
        iou = _compute_iou(new_box, tbox)
        if iou > best_iou:
            best_iou = iou
            best_id  = tid
    return best_id


def _get_stable_class(obj_id, new_class):
    """Vote on the most frequent class seen for this object over the last 5 frames."""
    _class_history[obj_id].append(new_class)
    hist = _class_history[obj_id]
    return max(set(hist), key=hist.count)


def _smooth_box(obj_id, box):
    """Average box coordinates over last 5 frames to eliminate jitter."""
    _box_history[obj_id].append(box)
    avg = [int(sum(v) / len(v)) for v in zip(*_box_history[obj_id])]
    return tuple(avg)  # (x1, y1, x2, y2)


def _reset_trackers():
    """Clear per-object histories between video streams."""
    _class_history.clear()
    _box_history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# STREAM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def stop_video_stream():
    global stop_vehicle_video
    stop_vehicle_video = True


def _draw_detections(frame, detections):
    """Overlay cached bounding boxes — zero inference cost."""
    for (x1, y1, x2, y2, cls_name, plate_rect, plate_text) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, cls_name, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if plate_rect is not None:
            px, py, pw, ph = plate_rect
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            label = plate_text if plate_text else "Plate"
            cv2.putText(frame, label, (px, max(py - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def _run_detection(frame, prev_tracks):
    """
    Run vehicle + plate inference with:
      - confidence filtering
      - class priority filtering for heavy vehicles
      - IoU-based ID assignment
      - class stabilization (majority vote)
      - box smoothing (running average)

    Returns (new_detections, new_tracks)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    results = vehicle_model(proc, imgsz=640, conf=0.4, verbose=False)[0]
    detections = []
    new_tracks  = {}  # obj_id → (x1,y1,x2,y2)

    # Track ID counter persists across calls via prev_tracks max key
    next_id = (max(prev_tracks.keys()) + 1) if prev_tracks else 0

    for box in results.boxes:
        raw_conf  = float(box.conf[0])
        cls_id    = int(box.cls[0])
        raw_class = vehicle_model.names[cls_id]

        if raw_class not in ['car', 'truck', 'bus', 'motorcycle']:
            continue

        # 1. CONFIDENCE FILTERING — drop weak predictions
        if raw_conf < 0.45:
            continue

        # 5. CLASS PRIORITY FILTER — heavy vehicles need higher confidence
        if raw_class in ('bus', 'truck') and raw_conf < 0.55:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        raw_box = (x1, y1, x2, y2)

        # 4. IOU TRACKING — match to existing track or create new one
        obj_id = _match_to_track(raw_box, prev_tracks)
        if obj_id is None:
            obj_id   = next_id
            next_id += 1

        # 2. CLASS STABILIZATION — majority vote over last 5 frames
        stable_class = _get_stable_class(obj_id, raw_class)

        # 3. BOX SMOOTHING — running average over last 5 frames
        sx1, sy1, sx2, sy2 = _smooth_box(obj_id, raw_box)

        new_tracks[obj_id] = (sx1, sy1, sx2, sy2)

        # Plate detection on smoothed crop
        plate_rect = None
        plate_text = ""

        if plate_model is not None:
            crop = proc[sy1:sy2, sx1:sx2]
            if crop.size > 0:
                pr = plate_model(crop, imgsz=320, conf=0.25, verbose=False)[0]
                for pbox in pr.boxes:
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                    plate_rect = (sx1 + px1, sy1 + py1, px2 - px1, py2 - py1)
                    plate_text, _ = extract_plate_text(proc, sx1, sy1, sx2 - sx1, sy2 - sy1)
                    break
        else:
            plate_text, plate_rect = extract_plate_text(proc, sx1, sy1, sx2 - sx1, sy2 - sy1)

        detections.append((sx1, sy1, sx2, sy2, stable_class, plate_rect, plate_text))

    return detections, new_tracks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN STREAMING GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_vehicle_frames(video_path):
    """MJPEG generator. Deletes *video_path* (temp file) when streaming ends."""
    global stop_vehicle_video
    stop_vehicle_video = False
    _reset_trackers()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = (1.0 / fps) if fps > 0 else 0.033

    frame_count     = 0
    last_detections = []
    prev_tracks     = {}  # obj_id → smoothed box (x1,y1,x2,y2)

    while True:
        if stop_vehicle_video:
            break

        t_start = time.time()
        success, frame = cap.read()
        if not success or frame is None:
            break

        frame_count += 1

        try:
            frame = cv2.resize(frame, (640, 480))

            # Run inference only every 3rd frame — big speedup, zero visible lag
            if frame_count % 3 == 0:
                last_detections, prev_tracks = _run_detection(frame, prev_tracks)

            _draw_detections(frame, last_detections)

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            elapsed = time.time() - t_start
            wait    = frame_delay - elapsed
            if wait > 0:
                time.sleep(wait)

        except Exception as e:
            print("Frame error:", e)
            continue

    cap.release()
    cv2.destroyAllWindows()

    # ── Delete temp file immediately after processing ────────────────────────
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"[vehicle_model] Deleted temp file: {video_path}")
    except Exception as e:
        print(f"[vehicle_model] Could not delete temp file {video_path}: {e}")
