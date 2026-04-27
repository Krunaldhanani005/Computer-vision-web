"""
fr_database.py  — Production-grade FR surveillance database layer
Database: fr_surveillance_db (MongoDB)

Collections:
    faces             – registered face encodings (known / blacklist)
    recognized_faces  – recognised person dashboard (known)
    unknown_faces     – unknown face dashboard
    blacklist_alerts  – blacklist alert gallery
    zones             – polygon zone storage
"""

import datetime
import os
from pymongo import MongoClient, DESCENDING

# ── Connection ────────────────────────────────────────────────────────────────
try:
    _client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    _db = _client["fr_surveillance_db"]

    faces_col        = _db["faces"]
    recognized_col   = _db["recognized_faces"]
    unknown_col      = _db["unknown_faces"]
    alerts_col       = _db["blacklist_alerts"]
    zones_col        = _db["zones"]

    # Indexes for performance
    faces_col.create_index("name")
    recognized_col.create_index([("last_seen", DESCENDING)])
    unknown_col.create_index([("last_seen", DESCENDING)])
    alerts_col.create_index([("alert_time", DESCENDING)])

    print("[fr_database] Connected to fr_surveillance_db ✓")
except Exception as e:
    print(f"[fr_database] MongoDB error: {e}")
    faces_col = recognized_col = unknown_col = alerts_col = zones_col = None


# ══════════════════════════════════════════════════════════════════════════════
#  FACES — Registration
# ══════════════════════════════════════════════════════════════════════════════

def insert_face(name: str, person_type: str, encoding: list) -> bool:
    """Store a face encoding (known / blacklist)."""
    if faces_col is None:
        return False
    try:
        faces_col.insert_one({
            "name":        name,
            "person_type": person_type,   # "known" | "blacklist"
            "encoding":    encoding,
            "created_at":  datetime.datetime.utcnow(),
            "status":      "active"
        })
        return True
    except Exception as e:
        print(f"[fr_database] insert_face error: {e}")
        return False


def get_all_faces():
    """Return all registered faces (name, person_type, encoding)."""
    if faces_col is None:
        return []
    return list(faces_col.find({}, {"_id": 0, "name": 1, "person_type": 1, "encoding": 1}))


def delete_all_faces() -> int:
    if faces_col is None:
        return 0
    return faces_col.delete_many({}).deleted_count


# ══════════════════════════════════════════════════════════════════════════════
#  RECOGNIZED FACES — Known person dashboard
# ══════════════════════════════════════════════════════════════════════════════

def upsert_recognized(name: str, confidence: float, snapshot_path: str):
    """Create or update a recognised person record (10 s cooldown enforced upstream)."""
    if recognized_col is None:
        return
    try:
        now = datetime.datetime.utcnow()
        existing = recognized_col.find_one({"name": name})
        if existing:
            recognized_col.update_one(
                {"name": name},
                {"$set":  {"last_seen": now, "confidence": confidence, "image_snapshot": snapshot_path},
                 "$inc":  {"total_detected": 1, "attendance_count": 1}}
            )
        else:
            recognized_col.insert_one({
                "name":             name,
                "person_type":      "known",
                "last_seen":        now,
                "first_seen":       now,
                "total_detected":   1,
                "attendance_count": 1,
                "image_snapshot":   snapshot_path,
                "confidence":       confidence
            })
    except Exception as e:
        print(f"[fr_database] upsert_recognized error: {e}")


def get_recognized_dashboard(limit: int = 50):
    if recognized_col is None:
        return []
    docs = list(recognized_col.find({}, {"_id": 0}).sort("last_seen", DESCENDING).limit(limit))
    for d in docs:
        if "last_seen" in d:
            d["last_seen"] = d["last_seen"].strftime("%Y-%m-%d %H:%M:%S")
        if "first_seen" in d:
            d["first_seen"] = d["first_seen"].strftime("%Y-%m-%d %H:%M:%S")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
#  UNKNOWN FACES — Unknown dashboard
# ══════════════════════════════════════════════════════════════════════════════

def upsert_unknown(temp_id: str, snapshot_path: str):
    """Create or bump an unknown-face record."""
    if unknown_col is None:
        return
    try:
        now = datetime.datetime.utcnow()
        existing = unknown_col.find_one({"temp_id": temp_id})
        if existing:
            unknown_col.update_one(
                {"temp_id": temp_id},
                {"$set": {"last_seen": now, "image_snapshot": snapshot_path},
                 "$inc": {"detection_count": 1}}
            )
        else:
            unknown_col.insert_one({
                "temp_id":        temp_id,
                "first_seen":     now,
                "last_seen":      now,
                "detection_count": 1,
                "image_snapshot": snapshot_path
            })
    except Exception as e:
        print(f"[fr_db] upsert_unknown error: {e}")

def delete_all_unknown_logs():
    """Delete all unknown person logs and clean up their static files."""
    if _logs_col is None:
        return False
    try:
        docs = _logs_col.find({"event": "unknown_person"})
        for d in docs:
            snap = d.get("snapshot")
            if snap:
                file_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", snap
                )
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
        _logs_col.delete_many({"event": "unknown_person"})
        return True
    except Exception as e:
        print(f"[fr_db] delete_all_unknown_logs error: {e}")
        return False


def get_unknown_dashboard(limit: int = 50):
    if unknown_col is None:
        return []
    docs = list(unknown_col.find({}, {"_id": 0}).sort("last_seen", DESCENDING).limit(limit))
    for d in docs:
        for k in ("first_seen", "last_seen"):
            if k in d and hasattr(d[k], "strftime"):
                d[k] = d[k].strftime("%Y-%m-%d %H:%M:%S")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
#  BLACKLIST ALERTS
# ══════════════════════════════════════════════════════════════════════════════

def insert_alert(name: str, snapshot_path: str, camera_source: str = "webcam", zone_id: str = "default"):
    if alerts_col is None:
        return
    try:
        alerts_col.insert_one({
            "name":          name,
            "snapshot":      snapshot_path,
            "alert_time":    datetime.datetime.utcnow(),
            "camera_source": camera_source,
            "zone_id":       zone_id
        })
    except Exception as e:
        print(f"[fr_database] insert_alert error: {e}")


def get_alerts_dashboard(limit: int = 50):
    if alerts_col is None:
        return []
    docs = list(alerts_col.find({}, {"_id": 0}).sort("alert_time", DESCENDING).limit(limit))
    for d in docs:
        if "alert_time" in d and hasattr(d["alert_time"], "strftime"):
            d["alert_time"] = d["alert_time"].strftime("%Y-%m-%d %H:%M:%S")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
#  ZONES — Polygon storage
# ══════════════════════════════════════════════════════════════════════════════

def save_polygon_zone(points: list, zone_id: str = "default") -> bool:
    """
    points: list of {x, y} normalised 0-1 dicts.
    Overwrites any existing zone with the same zone_id.
    """
    if zones_col is None:
        return False
    try:
        zones_col.delete_many({"zone_id": zone_id})
        zones_col.insert_one({
            "zone_id":    zone_id,
            "points":     points,
            "created_at": datetime.datetime.utcnow()
        })
        return True
    except Exception as e:
        print(f"[fr_database] save_polygon_zone error: {e}")
        return False


def load_polygon_zone(zone_id: str = "default"):
    """Returns list of normalised {x,y} points, or None."""
    if zones_col is None:
        return None
    doc = zones_col.find_one({"zone_id": zone_id})
    if doc and "points" in doc:
        return doc["points"]
    return None


def delete_polygon_zone(zone_id: str = "default"):
    if zones_col is None:
        return
    zones_col.delete_many({"zone_id": zone_id})
