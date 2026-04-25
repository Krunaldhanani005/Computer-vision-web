from pymongo import MongoClient

# Use the same MongoDB connection pattern as restricted_area database
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["computer_vision_db"]
    zones_collection = db["zones"]
except Exception as e:
    print(f"MongoDB connection error: {e}")
    zones_collection = None

class ZoneManager:
    def __init__(self):
        self.active_zone = self._load_zone_from_db()

    def validate_coordinates(self, zone):
        with open("zone_debug.txt", "a") as f:
            f.write(f"Validating zone: {zone}\n")
        if not isinstance(zone, dict):
            return False
        required_keys = ['x', 'y', 'w', 'h']
        if not all(k in zone for k in required_keys):
            return False
        try:
            if float(zone['w']) <= 0 or float(zone['h']) <= 0:
                with open("zone_debug.txt", "a") as f:
                    f.write(f"Failed dims check: {zone}\n")
                return False
        except Exception as e:
            with open("zone_debug.txt", "a") as f:
                f.write(f"Exception: {e}\n")
            return False
        return True

    def save_zone(self, zone_data):
        if not self.validate_coordinates(zone_data):
            return False

        if zones_collection is not None:
            # Strictly enforce a single active zone document
            zones_collection.delete_many({})
            zones_collection.insert_one({"zone": zone_data})
        self.active_zone = zone_data
        return True

    def load_zone(self):
        # Return from cache to avoid blocking DB query in the fast video loop
        return self.active_zone

    def _load_zone_from_db(self):
        if zones_collection is not None:
            doc = zones_collection.find_one({})
            if doc and "zone" in doc:
                return doc["zone"]
        return None

    def clear_zone(self):
        if zones_collection is not None:
            zones_collection.delete_many({})
        self.active_zone = None

    def is_face_inside_zone(self, face_rect, zone=None):
        """
        Check if the center of the face is inside the zone.
        face_rect: (x, y, w, h)
        zone: dict with 'x', 'y', 'w', 'h'
        """
        if zone is None:
            return True # If no zone is defined, everywhere is inside
        
        fx, fy, fw, fh = face_rect
        center_x = fx + fw / 2
        center_y = fy + fh / 2

        zx = zone['x']
        zy = zone['y']
        zw = zone['w']
        zh = zone['h']

        if zx <= center_x <= zx + zw and zy <= center_y <= zy + zh:
            return True
        return False

zone_manager = ZoneManager()
