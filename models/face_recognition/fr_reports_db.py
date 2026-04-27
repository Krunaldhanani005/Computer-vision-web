"""
fr_reports_db.py  —  Production FR Report / Dashboard database layer
Database  : fr_surveillance_db
Collections:
    known_reports      – one doc per registered person (profile + stats)
    attendance_reports – one doc per person per calendar day
    unknown_reports    – one doc per unique unknown temp_id
    blacklist_reports  – one doc per alert event
"""

import datetime
import os
import pytz
from pymongo import MongoClient, DESCENDING, ASCENDING

IST = pytz.timezone('Asia/Kolkata')

def to_ist(dt_utc) -> datetime.datetime:
    if not isinstance(dt_utc, datetime.datetime):
        return dt_utc
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=pytz.utc)
    return dt_utc.astimezone(IST)

# ── Connection (reuse same database as fr_database.py) ────────────────────────
try:
    _client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    _db = _client["fr_surveillance_db"]

    known_rep_col      = _db["known_reports"]
    attendance_col     = _db["attendance_reports"]
    unknown_rep_col    = _db["unknown_reports"]
    blacklist_rep_col  = _db["blacklist_reports"]

    # ── Indexes ───────────────────────────────────────────────────────────────
    known_rep_col.create_index("name", unique=True)
    known_rep_col.create_index([("last_seen", DESCENDING)])
    attendance_col.create_index([("name", ASCENDING), ("date", ASCENDING)], unique=True)
    unknown_rep_col.create_index("temp_id", unique=True)
    unknown_rep_col.create_index([("last_seen", DESCENDING)])
    blacklist_rep_col.create_index([("alert_time", DESCENDING)])
    blacklist_rep_col.create_index("name")

    print("[fr_reports_db] Connected to fr_surveillance_db ✓")
except Exception as e:
    print(f"[fr_reports_db] MongoDB error: {e}")
    known_rep_col = attendance_col = unknown_rep_col = blacklist_rep_col = None


# ── Snapshot storage dir (static/fr_reports/) ─────────────────────────────────
_REPORT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "static", "fr_reports"
)
os.makedirs(_REPORT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  KNOWN PERSON REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def upsert_known_report(name: str, person_type: str, snapshot_path: str, confidence: float):
    """
    Create or update a known-person report record.
    Called once per cooldown cycle from the recognition pipeline.
    """
    if known_rep_col is None:
        return
    try:
        now  = datetime.datetime.utcnow()
        date_str = now.strftime("%Y-%m-%d")

        existing = known_rep_col.find_one({"name": name})
        if existing:
            # Update: bump count, update last_seen, add snapshot (cap at 10)
            snaps = existing.get("snapshots", [])
            if snapshot_path and snapshot_path not in snaps:
                snaps = (snaps + [snapshot_path])[-10:]   # keep last 10
            known_rep_col.update_one(
                {"name": name},
                {"$set":  {
                    "last_seen":       now,
                    "latest_snapshot": snapshot_path or existing.get("latest_snapshot", ""),
                    "snapshots":       snaps,
                    "confidence":      confidence,
                },
                 "$inc":  {"total_detections": 1},
                 "$addToSet": {"attendance_days": date_str}}
            )
        else:
            known_rep_col.insert_one({
                "name":            name,
                "person_type":     person_type,
                "registered_at":   now,
                "last_seen":       now,
                "total_detections":1,
                "confidence":      confidence,
                "latest_snapshot": snapshot_path,
                "snapshots":       [snapshot_path] if snapshot_path else [],
                "attendance_days": [date_str],
            })
    except Exception as e:
        print(f"[fr_reports_db] upsert_known_report error: {e}")


def upsert_attendance(name: str, snapshot_path: str):
    """
    Mark attendance for *name* today.
    One document per (name, date). Duplicate calls on the same day → $inc only.
    """
    if attendance_col is None:
        return
    try:
        now      = datetime.datetime.utcnow()
        date_str = now.strftime("%Y-%m-%d")
        attendance_col.update_one(
            {"name": name, "date": date_str},
            {"$setOnInsert": {"first_seen_today": now, "snapshot": snapshot_path},
             "$set":         {"last_seen_today":  now},
             "$inc":         {"detection_count":  1}},
            upsert=True
        )
    except Exception as e:
        print(f"[fr_reports_db] upsert_attendance error: {e}")


def get_known_dashboard(search: str = "", sort_by: str = "last_seen",
                        page: int = 1, per_page: int = 24):
    """Return paginated known-person dashboard data with attendance stats."""
    if known_rep_col is None:
        return [], 0

    query = {}
    if search:
        query["name"] = {"$regex": search, "$options": "i"}

    sort_map = {
        "last_seen":        [("last_seen", DESCENDING)],
        "name_asc":         [("name", ASCENDING)],
        "name_desc":        [("name", DESCENDING)],
        "detections":       [("total_detections", DESCENDING)],
    }
    sort_key = sort_map.get(sort_by, [("last_seen", DESCENDING)])

    total = known_rep_col.count_documents(query)
    skip  = (page - 1) * per_page

    docs = list(known_rep_col.find(query, {"encoding": 0, "_id": 0})
                              .sort(sort_key)
                              .skip(skip)
                              .limit(per_page))

    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    for d in docs:
        att_days    = d.get("attendance_days", [])
        present_cnt = len(att_days)
        # Absent = registered days since first seen that have no detection
        registered  = d.get("registered_at", datetime.datetime.utcnow())
        total_days  = max(1, (datetime.datetime.utcnow() - registered).days + 1)
        absent_cnt  = max(0, total_days - present_cnt)
        att_pct     = round((present_cnt / total_days) * 100) if total_days > 0 else 0

        d["present_count"] = present_cnt
        d["absent_count"]  = absent_cnt
        d["attendance_pct"]= att_pct
        d["is_present_today"] = today in att_days
        d["total_snapshots"]  = len(d.get("snapshots", []))

        # Serialise datetimes to IST
        if "last_seen" in d and hasattr(d["last_seen"], "strftime"):
            ist_dt = to_ist(d["last_seen"])
            d["last_seen"] = ist_dt.strftime("%d-%m-%Y %H:%M:%S IST")
        
        if "registered_at" in d and hasattr(d["registered_at"], "strftime"):
            ist_dt = to_ist(d["registered_at"])
            d["first_seen"] = ist_dt.strftime("%d-%m-%Y %H:%M:%S IST")

    return docs, total


def get_known_stats():
    """Top metrics bar for Known dashboard."""
    if known_rep_col is None:
        return {}
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    registered   = known_rep_col.count_documents({})
    recog_today  = known_rep_col.count_documents({"attendance_days": today})
    # Average attendance percentage
    pipeline = [
        {"$project": {
            "att_days":   {"$size": {"$ifNull": ["$attendance_days", []]}},
            "registered_at": 1
        }},
        {"$group": {"_id": None, "total_att": {"$sum": "$att_days"}, "count": {"$sum": 1}}}
    ]
    agg  = list(known_rep_col.aggregate(pipeline))
    avg_att = 0
    if agg and agg[0]["count"] > 0:
        avg_att = round(agg[0]["total_att"] / max(agg[0]["count"], 1))

    return {
        "registered_faces": registered,
        "recognized_today": recog_today,
        "avg_attendance":   avg_att,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  UNKNOWN PERSON REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def upsert_unknown_report(temp_id: str, snapshot_path: str):
    """
    Create or update an unknown-face report record.
    Stores up to 5 snapshots per temp_id.
    """
    if unknown_rep_col is None:
        return
    try:
        now = datetime.datetime.utcnow()
        existing = unknown_rep_col.find_one({"temp_id": temp_id})
        if existing:
            snaps = existing.get("snapshots", [])
            if snapshot_path and snapshot_path not in snaps:
                snaps = (snaps + [snapshot_path])[-5:]
            unknown_rep_col.update_one(
                {"temp_id": temp_id},
                {"$set":  {"last_seen": now, "latest_snapshot": snapshot_path, "snapshots": snaps},
                 "$inc":  {"detection_count": 1}}
            )
        else:
            unknown_rep_col.insert_one({
                "temp_id":         temp_id,
                "first_seen":      now,
                "last_seen":       now,
                "detection_count": 1,
                "latest_snapshot": snapshot_path,
                "snapshots":       [snapshot_path] if snapshot_path else [],
            })
    except Exception as e:
        print(f"[fr_reports_db] upsert_unknown_report error: {e}")


def get_unknown_dashboard(search: str = "", sort_by: str = "last_seen",
                          page: int = 1, per_page: int = 50):
    if unknown_rep_col is None:
        return [], 0

    query = {}
    if search:
        query["temp_id"] = {"$regex": search, "$options": "i"}

    sort_map = {
        "last_seen":   [("last_seen", DESCENDING)],
        "first_seen":  [("first_seen", DESCENDING)],
        "detections":  [("detection_count", DESCENDING)],
    }
    sort_key = sort_map.get(sort_by, [("last_seen", DESCENDING)])

    total = unknown_rep_col.count_documents(query)
    skip  = (page - 1) * per_page
    docs  = list(unknown_rep_col.find(query, {"_id": 0})
                                .sort(sort_key)
                                .skip(skip)
                                .limit(per_page))

    for d in docs:
        if "last_seen" in d and hasattr(d["last_seen"], "strftime"):
            ist_dt = to_ist(d["last_seen"])
            d["date"] = ist_dt.strftime("%d-%m-%Y")
            d["time"] = ist_dt.strftime("%H:%M:%S IST")

    return docs, total

def delete_all_unknown_reports():
    """Delete all unknown reports from DB and clean up their static files."""
    if unknown_rep_col is None:
        return False
    try:
        # Delete static files
        docs = unknown_rep_col.find({})
        for d in docs:
            for snap in d.get("snapshots", []):
                file_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", snap
                )
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
        # Delete from DB
        unknown_rep_col.delete_many({})
        return True
    except Exception as e:
        print(f"[fr_reports_db] Error deleting unknown reports: {e}")
        return False


def get_unknown_stats():
    if unknown_rep_col is None:
        return {}
    today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        "total_unknown":   unknown_rep_col.count_documents({}),
        "unknown_today":   unknown_rep_col.count_documents({"first_seen": {"$gte": today_start}}),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BLACKLIST REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def insert_blacklist_report(name: str, snapshot_path: str,
                            camera_source: str = "webcam", zone_id: str = "default"):
    """One document per alert event — gallery-style, many rows per person."""
    if blacklist_rep_col is None:
        return
    try:
        blacklist_rep_col.insert_one({
            "name":          name,
            "snapshot":      snapshot_path,
            "alert_time":    datetime.datetime.utcnow(),
            "camera_source": camera_source,
            "zone_id":       zone_id,
        })
    except Exception as e:
        print(f"[fr_reports_db] insert_blacklist_report error: {e}")


def get_blacklist_dashboard(search: str = "", date_filter: str = "",
                            page: int = 1, per_page: int = 50):
    if blacklist_rep_col is None:
        return [], 0

    query = {}
    if search:
        query["name"] = {"$regex": search, "$options": "i"}
    if date_filter:
        try:
            day_start = datetime.datetime.strptime(date_filter, "%Y-%m-%d")
            day_end   = day_start + datetime.timedelta(days=1)
            query["alert_time"] = {"$gte": day_start, "$lt": day_end}
        except ValueError:
            pass

    total = blacklist_rep_col.count_documents(query)
    skip  = (page - 1) * per_page
    docs  = list(blacklist_rep_col.find(query, {"_id": 0})
                                  .sort("alert_time", DESCENDING)
                                  .skip(skip)
                                  .limit(per_page))

    for d in docs:
        if "alert_time" in d and hasattr(d["alert_time"], "strftime"):
            ist_dt = to_ist(d["alert_time"])
            d["alert_time"] = ist_dt.strftime("%d-%m-%Y %H:%M:%S IST")

    return docs, total


def get_blacklist_stats():
    if blacklist_rep_col is None:
        return {}
    today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        "total_alerts":   blacklist_rep_col.count_documents({}),
        "alerts_today":   blacklist_rep_col.count_documents({"alert_time": {"$gte": today_start}}),
        "unique_persons": len(blacklist_rep_col.distinct("name")),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def export_known_csv() -> str:
    """Return CSV string of all known persons."""
    if known_rep_col is None:
        return "name,person_type,last_seen,present_count,attendance_pct,total_detections\n"
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    rows  = ["name,person_type,last_seen,present_count,absent_count,attendance_pct,total_detections,is_present_today"]
    for d in known_rep_col.find({}, {"_id": 0, "snapshots": 0, "encoding": 0}):
        att_days = d.get("attendance_days", [])
        present  = len(att_days)
        reg      = d.get("registered_at", datetime.datetime.utcnow())
        days     = max(1, (datetime.datetime.utcnow() - reg).days + 1)
        absent   = max(0, days - present)
        pct      = round((present / days) * 100)
        last     = d.get("last_seen", "")
        if hasattr(last, "strftime"):
            ist_dt = to_ist(last)
            last = ist_dt.strftime("%d-%m-%Y %H:%M:%S IST")
        rows.append(f"{d.get('name','')},{d.get('person_type','')},{last},{present},{absent},{pct},{d.get('total_detections',0)},{today in att_days}")
    return "\n".join(rows)


def export_blacklist_csv() -> str:
    if blacklist_rep_col is None:
        return "name,alert_time,camera_source,zone_id\n"
    rows = ["name,alert_time,camera_source,zone_id"]
    for d in blacklist_rep_col.find({}, {"_id": 0, "snapshot": 0}):
        t = d.get("alert_time", "")
        if hasattr(t, "strftime"):
            ist_dt = to_ist(t)
            t = ist_dt.strftime("%d-%m-%Y %H:%M:%S IST")
        rows.append(f"{d.get('name','')},{t},{d.get('camera_source','')},{d.get('zone_id','')}")
    return "\n".join(rows)
