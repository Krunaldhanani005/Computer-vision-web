"""
models/restricted_area/database.py
────────────────────────────────────
Independent MongoDB storage for Restricted Area module.
Database : restricted_area_db  (SEPARATE from fr_surveillance_db)

Collections:
    ra_known_persons  — authorised personnel ArcFace encodings
    ra_events         — intrusion event log (one per unique unknown slot)
    ra_alerts         — individual alert records
    ra_snapshots      — snapshot file paths per event
    ra_zones          — zone storage (separate from FR zones)
"""

import datetime
import io
import csv
import numpy as np
import pytz
from pymongo import MongoClient, DESCENDING

IST = pytz.timezone('Asia/Kolkata')


def to_ist(dt_utc):
    if not isinstance(dt_utc, datetime.datetime):
        return dt_utc
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=pytz.utc)
    return dt_utc.astimezone(IST)


try:
    _client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    _db = _client["restricted_area_db"]

    ra_known_persons = _db["ra_known_persons"]
    ra_events        = _db["ra_events"]
    ra_alerts        = _db["ra_alerts"]
    ra_snapshots     = _db["ra_snapshots"]
    ra_zones         = _db["ra_zones"]

    # Indexes
    ra_known_persons.create_index("name", unique=True)
    ra_events.create_index("event_id", unique=True)
    ra_events.create_index([("first_seen", DESCENDING)])
    ra_alerts.create_index([("alert_time", DESCENDING)])
    ra_snapshots.create_index("event_id")

    print("[restricted_area.db] Connected to restricted_area_db ✓")
except Exception as e:
    print(f"[restricted_area.db] MongoDB error: {e}")
    ra_known_persons = ra_events = ra_alerts = ra_snapshots = ra_zones = None



# ══════════════════════════════════════════════════════════════════════════════
#  ZONE STORAGE  (separate from FR zones)
# ══════════════════════════════════════════════════════════════════════════════

_ra_zone_cache = None
_ra_zone_loaded = False

def save_ra_zone(points: list, zone_id: str = "restricted_default") -> bool:
    """Upsert a polygon zone for Restricted Area."""
    global _ra_zone_cache, _ra_zone_loaded
    if ra_zones is None:
        return False
    try:
        ra_zones.update_one(
            {"zone_id": zone_id},
            {"$set": {"zone_id": zone_id, "points": points,
                      "updated_at": datetime.datetime.utcnow()}},
            upsert=True
        )
        _ra_zone_cache = points
        _ra_zone_loaded = True
        return True
    except Exception as e:
        print(f"[restricted_area.db] save_ra_zone error: {e}")
        return False


def clear_ra_zone(zone_id: str = "restricted_default"):
    """Delete a polygon zone for Restricted Area."""
    global _ra_zone_cache, _ra_zone_loaded
    if ra_zones is not None:
        try:
            ra_zones.delete_many({"zone_id": zone_id})
        except Exception as e:
            print(f"[restricted_area.db] clear_ra_zone error: {e}")
    _ra_zone_cache = None
    _ra_zone_loaded = True


def load_ra_zone(zone_id: str = "restricted_default", force: bool = False):
    """Load zone points for Restricted Area. Returns list of {x,y} dicts or None."""
    global _ra_zone_cache, _ra_zone_loaded
    if not force and _ra_zone_loaded:
        return _ra_zone_cache

    if ra_zones is None:
        return None
    try:
        doc = ra_zones.find_one({"zone_id": zone_id})
        if doc and doc.get("points"):
            _ra_zone_cache = doc["points"]
        else:
            _ra_zone_cache = None
        _ra_zone_loaded = True
        return _ra_zone_cache
    except Exception as e:
        print(f"[restricted_area.db] load_ra_zone error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  KNOWN PERSONS (authorised — those allowed in restricted zone)
# ══════════════════════════════════════════════════════════════════════════════

def insert_ra_known_person(name: str, encoding: list) -> bool:
    """Store one ArcFace embedding for an authorised person."""
    if ra_known_persons is None:
        return False
    try:
        now = datetime.datetime.utcnow()
        existing = ra_known_persons.find_one({"name": name})
        if existing:
            ra_known_persons.update_one(
                {"name": name},
                {"$push": {"encodings": encoding}, "$set": {"updated_at": now}}
            )
        else:
            ra_known_persons.insert_one({
                "name": name,
                "encodings": [encoding],
                "created_at": now,
                "updated_at": now
            })
        return True
    except Exception as e:
        print(f"[restricted_area.db] insert_known error: {e}")
        return False


def load_all_ra_known_persons() -> tuple:
    """Returns (names: list[str], encodings: list[np.ndarray])."""
    names, encodings = [], []
    if ra_known_persons is None:
        return names, encodings
    try:
        for doc in ra_known_persons.find({}, {"_id": 0, "name": 1, "encodings": 1}):
            for enc in doc.get("encodings", []):
                names.append(doc["name"])
                encodings.append(np.array(enc, dtype=np.float32))
    except Exception as e:
        print(f"[restricted_area.db] load_known error: {e}")
    return names, encodings


# Legacy alias used by old __init__.py
def load_all_known_persons():
    return load_all_ra_known_persons()


# ══════════════════════════════════════════════════════════════════════════════
#  EVENTS & ALERTS
# ══════════════════════════════════════════════════════════════════════════════

def upsert_ra_event(event_id: str, camera_source: str, zone_id: str, snapshot_path: str,
                    person_type: str = "unknown", person_name: str = "Unknown"):
    """Upsert a RA intrusion event (grouped by slot event_id)."""
    if ra_events is None:
        return
    now = datetime.datetime.utcnow()
    try:
        ra_events.update_one(
            {"event_id": event_id},
            {
                "$setOnInsert": {
                    "first_seen":    now,
                    "camera_source": camera_source,
                    "zone_id":       zone_id,
                    "person_type":   person_type,
                    "person_name":   person_name,
                },
                "$set": {"last_seen": now, "latest_snapshot": snapshot_path},
                "$inc": {"detection_count": 1}
            },
            upsert=True
        )
        if snapshot_path and ra_snapshots is not None:
            ra_snapshots.insert_one({
                "event_id":      event_id,
                "snapshot_path": snapshot_path,
                "timestamp":     now,
            })
    except Exception as e:
        print(f"[restricted_area.db] upsert_event error: {e}")


def insert_ra_alert(event_id: str, snapshot_path: str, camera_source: str, zone_id: str,
                    person_type: str = "unknown", person_name: str = "Unknown"):
    """Insert one RA alert record."""
    if ra_alerts is None:
        return
    try:
        ra_alerts.insert_one({
            "event_id":      event_id,
            "snapshot_path": snapshot_path,
            "alert_time":    datetime.datetime.utcnow(),
            "camera_source": camera_source,
            "zone_id":       zone_id,
            "person_type":   person_type,
            "person_name":   person_name,
        })
    except Exception as e:
        print(f"[restricted_area.db] insert_alert error: {e}")


# Legacy alias used by old __init__.py
def log_alert(image_path: str, camera_source: str = "webcam",
              zone_id: str = "Default", event_id: str = ""):
    insert_ra_alert(event_id, image_path, camera_source, zone_id)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD QUERIES
# ══════════════════════════════════════════════════════════════════════════════

def get_ra_dashboard(search: str = "", date_filter: str = "",
                     page: int = 1, per_page: int = 50):
    if ra_events is None:
        return [], 0
    query = {}
    if search:
        query["$or"] = [
            {"camera_source": {"$regex": search, "$options": "i"}},
            {"zone_id":       {"$regex": search, "$options": "i"}},
            {"event_id":      {"$regex": search, "$options": "i"}},
        ]
    if date_filter:
        try:
            start = datetime.datetime.strptime(date_filter, "%Y-%m-%d")
            end   = start + datetime.timedelta(days=1)
            query["first_seen"] = {"$gte": start, "$lt": end}
        except Exception:
            pass

    total = ra_events.count_documents(query)
    docs  = list(
        ra_events.find(query, {"_id": 0})
        .sort("first_seen", DESCENDING)
        .skip((page - 1) * per_page)
        .limit(per_page)
    )
    for d in docs:
        for key in ("first_seen", "last_seen"):
            if key in d and hasattr(d[key], "strftime"):
                d[key] = to_ist(d[key]).strftime("%d-%m-%Y %H:%M:%S IST")
    return docs, total


# Legacy alias used by app.py
def get_restricted_dashboard(search: str = "", date_filter: str = "",
                              page: int = 1, per_page: int = 50):
    return get_ra_dashboard(search, date_filter, page, per_page)


def get_ra_stats() -> dict:
    if ra_events is None:
        return {}
    today_start = datetime.datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return {
        "total_intrusions": ra_events.count_documents({}),
        "intrusions_today": ra_events.count_documents(
            {"first_seen": {"$gte": today_start}}
        ),
        "total_alerts": ra_alerts.count_documents({}) if ra_alerts else 0,
        "alerts_today": (
            ra_alerts.count_documents({"alert_time": {"$gte": today_start}})
            if ra_alerts else 0
        ),
    }


def export_ra_csv() -> str:
    docs, _ = get_ra_dashboard("", "", 1, 100000)
    output  = io.StringIO()
    writer  = csv.writer(output)
    writer.writerow([
        "Event ID", "First Seen", "Last Seen",
        "Camera Source", "Zone", "Detections", "Snapshot"
    ])
    for d in docs:
        writer.writerow([
            d.get("event_id", ""),
            d.get("first_seen", ""),
            d.get("last_seen", ""),
            d.get("camera_source", ""),
            d.get("zone_id", ""),
            d.get("detection_count", 0),
            d.get("latest_snapshot", ""),
        ])
    return output.getvalue()


# Legacy alias
def export_restricted_csv() -> str:
    return export_ra_csv()
