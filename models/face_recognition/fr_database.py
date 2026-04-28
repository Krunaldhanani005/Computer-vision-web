"""
fr_database.py — Unified Database for Face Recognition
Database: fr_surveillance_db

Collections:
    known_persons      – Registered known faces + encodings + dashboard stats
    known_attendance   – Attendance records per day per known person
    unknown_persons    – Unknown face trackers and stats
    unknown_snapshots  – Snapshots for unknown faces
    blacklist_persons  – Registered blacklist faces + encodings + stats
    blacklist_alerts   – Individual alert events for blacklist detections
    zones              – Polygon zone storage
"""

import datetime
import pytz
from pymongo import MongoClient, DESCENDING, ASCENDING

IST = pytz.timezone('Asia/Kolkata')

def to_ist(dt_utc) -> datetime.datetime:
    if not isinstance(dt_utc, datetime.datetime):
        return dt_utc
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=pytz.utc)
    return dt_utc.astimezone(IST)

try:
    _client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    _db = _client["fr_surveillance_db"]

    known_persons     = _db["known_persons"]
    known_attendance  = _db["known_attendance"]
    unknown_persons   = _db["unknown_persons"]
    unknown_snapshots = _db["unknown_snapshots"]
    blacklist_persons = _db["blacklist_persons"]
    blacklist_alerts  = _db["blacklist_alerts"]
    zones_col         = _db["zones"]

    # Indexes
    known_persons.create_index("name", unique=True)
    known_persons.create_index([("last_seen", DESCENDING)])
    
    known_attendance.create_index([("name", ASCENDING), ("date", ASCENDING)], unique=True)
    
    unknown_persons.create_index("temp_id", unique=True)
    unknown_persons.create_index([("last_seen", DESCENDING)])
    
    unknown_snapshots.create_index("temp_id")
    unknown_snapshots.create_index([("timestamp", DESCENDING)])

    blacklist_persons.create_index("name", unique=True)
    
    blacklist_alerts.create_index([("alert_time", DESCENDING)])
    blacklist_alerts.create_index("name")

    print("[fr_database] Connected to fr_surveillance_db ✓")
except Exception as e:
    print(f"[fr_database] MongoDB error: {e}")
    known_persons = known_attendance = unknown_persons = unknown_snapshots = blacklist_persons = blacklist_alerts = zones_col = None

# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRATION & ENCODINGS
# ══════════════════════════════════════════════════════════════════════════════

def insert_face(name: str, person_type: str, encoding: list) -> bool:
    try:
        now = datetime.datetime.utcnow()
        if person_type == "blacklist":
            col = blacklist_persons
        else:
            col = known_persons

        existing = col.find_one({"name": name})
        if existing:
            # Update encoding
            col.update_one({"name": name}, {"$set": {"encoding": encoding}})
        else:
            col.insert_one({
                "name": name,
                "person_type": person_type,
                "encoding": encoding,
                "created_at": now,
                "status": "active",
                "total_detections": 0
            })
        return True
    except Exception as e:
        print(f"[fr_database] insert_face error: {e}")
        return False

def get_all_faces():
    faces = []
    if known_persons is not None:
        for doc in known_persons.find({}, {"_id": 0, "name": 1, "person_type": 1, "encoding": 1}):
            faces.append(doc)
    if blacklist_persons is not None:
        for doc in blacklist_persons.find({}, {"_id": 0, "name": 1, "person_type": 1, "encoding": 1}):
            faces.append(doc)
    return faces

def delete_all_faces() -> int:
    count = 0
    if known_persons is not None:
        count += known_persons.delete_many({}).deleted_count
    if blacklist_persons is not None:
        count += blacklist_persons.delete_many({}).deleted_count
    return count

# ══════════════════════════════════════════════════════════════════════════════
#  KNOWN PERSONS & ATTENDANCE
# ══════════════════════════════════════════════════════════════════════════════

def upsert_known(name: str, confidence: float, snapshot_path: str):
    if known_persons is None: return
    now = datetime.datetime.utcnow()
    date_str = to_ist(now).strftime("%Y-%m-%d")

    known_persons.update_one(
        {"name": name},
        {
            "$set": {
                "last_seen":       now,
                "latest_snapshot": snapshot_path,
                "confidence":      confidence,
            },
            "$inc": {"total_detections": 1},
            "$addToSet": {"attendance_days": date_str},
            "$setOnInsert": {
                "created_at":  now,
                "person_type": "known",
                "status":      "active",
            },
        },
        upsert=True,
    )

    if known_attendance is not None:
        known_attendance.update_one(
            {"name": name, "date": date_str},
            {
                "$setOnInsert": {"first_seen_today": now},
                "$set": {"last_seen_today": now, "snapshot": snapshot_path},
                "$inc": {"detection_count": 1}
            },
            upsert=True
        )

def get_known_dashboard(search: str = "", sort_by: str = "last_seen", page: int = 1, per_page: int = 24):
    if known_persons is None: return [], 0
    query = {}
    if search: query["name"] = {"$regex": search, "$options": "i"}

    sort_map = {
        "last_seen": [("last_seen", DESCENDING)],
        "name_asc": [("name", ASCENDING)],
        "name_desc": [("name", DESCENDING)],
        "detections": [("total_detections", DESCENDING)],
    }
    sort_key = sort_map.get(sort_by, [("last_seen", DESCENDING)])

    total = known_persons.count_documents(query)
    skip = (page - 1) * per_page
    docs = list(known_persons.find(query, {"encoding": 0, "_id": 0}).sort(sort_key).skip(skip).limit(per_page))

    today = to_ist(datetime.datetime.utcnow()).strftime("%Y-%m-%d")
    for d in docs:
        att_days = d.get("attendance_days", [])
        d["present_count"] = len(att_days)
        d["is_present_today"] = today in att_days
        if "last_seen" in d and hasattr(d["last_seen"], "strftime"):
            d["last_seen"] = to_ist(d["last_seen"]).strftime("%d-%m-%Y %H:%M:%S IST")
        if "created_at" in d and hasattr(d["created_at"], "strftime"):
            d["first_seen"] = to_ist(d["created_at"]).strftime("%d-%m-%Y %H:%M:%S IST")

    return docs, total

def get_known_stats():
    if known_persons is None: return {}
    today = to_ist(datetime.datetime.utcnow()).strftime("%Y-%m-%d")
    return {
        "registered_faces": known_persons.count_documents({}),
        "recognized_today": known_persons.count_documents({"attendance_days": today}),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  UNKNOWN PERSONS & SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════════════

def upsert_unknown(temp_id: str, snapshot_path: str):
    if unknown_persons is None: return
    now = datetime.datetime.utcnow()
    
    unknown_persons.update_one(
        {"temp_id": temp_id},
        {
            "$setOnInsert": {"first_seen": now},
            "$set": {"last_seen": now, "latest_snapshot": snapshot_path},
            "$inc": {"detection_count": 1}
        },
        upsert=True
    )
    
    if unknown_snapshots is not None and snapshot_path:
        unknown_snapshots.insert_one({
            "temp_id": temp_id,
            "snapshot_path": snapshot_path,
            "timestamp": now
        })

def get_unknown_dashboard(search: str = "", sort_by: str = "last_seen", page: int = 1, per_page: int = 50):
    if unknown_persons is None: return [], 0
    query = {}
    if search: query["temp_id"] = {"$regex": search, "$options": "i"}

    sort_map = {
        "last_seen": [("last_seen", DESCENDING)],
        "first_seen": [("first_seen", DESCENDING)],
        "detections": [("detection_count", DESCENDING)],
    }
    sort_key = sort_map.get(sort_by, [("last_seen", DESCENDING)])

    total = unknown_persons.count_documents(query)
    skip = (page - 1) * per_page
    docs = list(unknown_persons.find(query, {"_id": 0}).sort(sort_key).skip(skip).limit(per_page))

    for d in docs:
        if "last_seen" in d and hasattr(d["last_seen"], "strftime"):
            ist_dt = to_ist(d["last_seen"])
            d["date"] = ist_dt.strftime("%d-%m-%Y")
            d["time"] = ist_dt.strftime("%H:%M:%S IST")

    return docs, total

def get_unknown_stats():
    if unknown_persons is None: return {}
    today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        "total_unknown": unknown_persons.count_documents({}),
        "unknown_today": unknown_persons.count_documents({"first_seen": {"$gte": today_start}}),
    }

def delete_all_unknown_logs():
    if unknown_persons is None: return False
    try:
        unknown_persons.delete_many({})
        unknown_snapshots.delete_many({})
        return True
    except:
        return False

# ══════════════════════════════════════════════════════════════════════════════
#  BLACKLIST ALERTS & PERSONS
# ══════════════════════════════════════════════════════════════════════════════

def insert_alert(name: str, snapshot_path: str, camera_source: str = "webcam", zone_id: str = "default"):
    if blacklist_alerts is None: return
    now = datetime.datetime.utcnow()
    
    # Also update blacklist_persons stats
    if blacklist_persons is not None:
        blacklist_persons.update_one(
            {"name": name},
            {"$set": {"last_seen": now, "latest_snapshot": snapshot_path}, "$inc": {"total_detections": 1}}
        )

    blacklist_alerts.insert_one({
        "name": name,
        "snapshot": snapshot_path,
        "alert_time": now,
        "camera_source": camera_source,
        "zone_id": zone_id
    })

def get_blacklist_dashboard(search: str = "", date_filter: str = "", page: int = 1, per_page: int = 50):
    if blacklist_alerts is None: return [], 0
    query = {}
    if search: query["name"] = {"$regex": search, "$options": "i"}
    if date_filter:
        try:
            day_start = datetime.datetime.strptime(date_filter, "%Y-%m-%d")
            day_end = day_start + datetime.timedelta(days=1)
            query["alert_time"] = {"$gte": day_start, "$lt": day_end}
        except ValueError:
            pass

    total = blacklist_alerts.count_documents(query)
    skip = (page - 1) * per_page
    docs = list(blacklist_alerts.find(query, {"_id": 0}).sort("alert_time", DESCENDING).skip(skip).limit(per_page))

    for d in docs:
        if "alert_time" in d and hasattr(d["alert_time"], "strftime"):
            d["alert_time"] = to_ist(d["alert_time"]).strftime("%d-%m-%Y %H:%M:%S IST")

    return docs, total

def get_blacklist_stats():
    if blacklist_alerts is None: return {}
    today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        "total_alerts": blacklist_alerts.count_documents({}),
        "alerts_today": blacklist_alerts.count_documents({"alert_time": {"$gte": today_start}}),
        "unique_persons": len(blacklist_alerts.distinct("name")),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def export_known_csv() -> str:
    if known_persons is None: return "name,last_seen,present_count,total_detections\n"
    rows = ["name,last_seen,present_count,total_detections"]
    for d in known_persons.find({}, {"_id": 0, "encoding": 0}):
        last = to_ist(d.get("last_seen", "")).strftime("%d-%m-%Y %H:%M:%S IST") if hasattr(d.get("last_seen"), "strftime") else ""
        rows.append(f"{d.get('name','')},{last},{len(d.get('attendance_days', []))},{d.get('total_detections',0)}")
    return "\n".join(rows)

def export_blacklist_csv() -> str:
    if blacklist_alerts is None: return "name,alert_time,camera_source,zone_id\n"
    rows = ["name,alert_time,camera_source,zone_id"]
    for d in blacklist_alerts.find({}, {"_id": 0}):
        t = to_ist(d.get("alert_time", "")).strftime("%d-%m-%Y %H:%M:%S IST") if hasattr(d.get("alert_time"), "strftime") else ""
        rows.append(f"{d.get('name','')},{t},{d.get('camera_source','')},{d.get('zone_id','')}")
    return "\n".join(rows)

# ══════════════════════════════════════════════════════════════════════════════
#  ZONES
# ══════════════════════════════════════════════════════════════════════════════

def save_polygon_zone(points: list, zone_id: str = "default") -> bool:
    if zones_col is None: return False
    try:
        zones_col.delete_many({"zone_id": zone_id})
        zones_col.insert_one({"zone_id": zone_id, "points": points, "created_at": datetime.datetime.utcnow()})
        return True
    except:
        return False

def load_polygon_zone(zone_id: str = "default"):
    if zones_col is None: return None
    doc = zones_col.find_one({"zone_id": zone_id})
    return doc["points"] if doc and "points" in doc else None

def delete_polygon_zone(zone_id: str = "default"):
    if zones_col is None: return
    zones_col.delete_many({"zone_id": zone_id})

