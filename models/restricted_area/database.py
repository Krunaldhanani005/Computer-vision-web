"""
models/restricted_area/database.py
───────────────────────────────────
All MongoDB interactions for the Restricted Area module.
Database : restricted_area_db
Collections:
    known_persons  — authorised personnel encodings
    alerts         — intrusion event log
"""

import numpy as np
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, PyMongoError

# ── Connection (singleton pool) ───────────────────────────────────────────────
_MONGO_URI = "mongodb://localhost:27017/"
_DB_NAME   = "restricted_area_db"

try:
    _client = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=3000)
    _client.server_info()            # force connection check
    _db = _client[_DB_NAME]
    known_persons_col = _db["known_persons"]
    alerts_col        = _db["alerts"]
    print("[restricted_area.db] Connected to MongoDB.")
except ConnectionFailure as e:
    print(f"[restricted_area.db] WARNING: MongoDB unavailable — {e}")
    _db = None
    known_persons_col = None
    alerts_col        = None


# ── Public helpers ─────────────────────────────────────────────────────────────

def load_all_known_persons() -> tuple[list, list]:
    """
    Load every document from known_persons and return
    (names: list[str], encodings: list[np.ndarray]).
    Safe to call even when DB is down — returns empty lists.
    """
    names, encodings = [], []
    if known_persons_col is None:
        return names, encodings
    try:
        for doc in known_persons_col.find({}, {"name": 1, "encoding": 1}):
            enc = np.array(doc["encoding"], dtype=np.float64)
            if enc.shape == (128,):           # guard against corrupt data
                names.append(doc["name"])
                encodings.append(enc)
    except PyMongoError as e:
        print(f"[restricted_area.db] load error: {e}")
    return names, encodings


def insert_known_person(name: str, encoding: np.ndarray) -> bool:
    """
    Persist one face encoding for an authorised person.
    Returns True on success.
    """
    if known_persons_col is None:
        return False
    try:
        known_persons_col.insert_one({
            "name":       name,
            "encoding":   encoding.tolist(),
            "created_at": datetime.utcnow(),
        })
        return True
    except PyMongoError as e:
        print(f"[restricted_area.db] insert error: {e}")
        return False


def log_alert(image_path: str, status: str = "unknown") -> None:
    """
    Append one intrusion event (called asynchronously — never blocks stream).
    """
    if alerts_col is None:
        return
    try:
        alerts_col.insert_one({
            "image_path": image_path,
            "timestamp":  datetime.utcnow(),
            "status":     status,
        })
    except PyMongoError as e:
        print(f"[restricted_area.db] alert log error: {e}")


def get_recent_alerts(limit: int = 20) -> list[dict]:
    """
    Fetch the N most-recent alert documents for the UI table.
    Returns a plain list of dicts (timestamp converted to ISO string).
    """
    if alerts_col is None:
        return []
    try:
        docs = alerts_col.find(
            {}, {"_id": 0, "image_path": 1, "timestamp": 1, "status": 1}
        ).sort("timestamp", DESCENDING).limit(limit)
        results = []
        for d in docs:
            d["timestamp"] = d["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            results.append(d)
        return results
    except PyMongoError as e:
        print(f"[restricted_area.db] fetch alerts error: {e}")
        return []
