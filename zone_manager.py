"""
zone_manager.py  — Polygon-based zone system

Zone is stored as a list of normalised {x, y} points (0.0–1.0).
Point-in-polygon test uses cv2.pointPolygonTest for accuracy.

Storage:
    Primary  : fr_surveillance_db.zones  (fr_database layer)
    Legacy   : computer_vision_db.zones  (kept for backwards compat, not used for FR)
"""

import cv2
import numpy as np

from models.face_recognition.fr_database import (
    save_polygon_zone,
    load_polygon_zone,
    delete_polygon_zone,
)


class ZoneManager:
    """
    Polygon zone manager.
    
    Zone format:  list of {"x": float, "y": float}  (normalised 0–1)
    """

    MIN_POINTS = 3  # need at least a triangle

    def __init__(self):
        self.active_points = self._load_from_db()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_from_db(self):
        pts = load_polygon_zone("default")
        if pts and len(pts) >= self.MIN_POINTS:
            return pts
        return None

    def save_zone(self, points: list) -> bool:
        """
        Save polygon zone.
        points: [{"x": ..., "y": ...}, ...]  normalised 0–1
        """
        if not points or len(points) < self.MIN_POINTS:
            return False
        if not all(isinstance(p, dict) and "x" in p and "y" in p for p in points):
            return False
        if save_polygon_zone(points, "default"):
            self.active_points = points
            return True
        return False

    def clear_zone(self):
        delete_polygon_zone("default")
        self.active_points = None

    def load_zone(self):
        """Returns active points list (cached) or None."""
        return self.active_points

    # ── Geometry ──────────────────────────────────────────────────────────────

    def to_pixel_polygon(self, points: list, frame_w: int, frame_h: int) -> np.ndarray:
        """Convert normalised points → pixel polygon (Nx1x2 int32 array for cv2)."""
        pts = np.array([[int(p["x"] * frame_w), int(p["y"] * frame_h)]
                        for p in points], dtype=np.int32)
        return pts.reshape((-1, 1, 2))

    def is_face_inside_zone(self, face_rect: tuple, points=None) -> bool:
        """
        Point-in-polygon test using the face centre.

        Args:
            face_rect: (x, y, w, h) in **pixel** coordinates
            points:    list of {"x", "y"} normalised dicts
                       — caller must pass the *pixel-scaled* polygon or normalise here.
                       If points is a dict (legacy rect format) we fall back to AABB.

        Returns True if inside zone or zone is None.
        """
        if points is None:
            return True

        # ── Legacy rectangle fallback (dict with x/y/w/h) ────────────────────
        if isinstance(points, dict):
            fx, fy, fw, fh = face_rect
            cx = fx + fw / 2
            cy = fy + fh / 2
            return (points["x"] <= cx <= points["x"] + points["w"] and
                    points["y"] <= cy <= points["y"] + points["h"])

        # ── Polygon path ──────────────────────────────────────────────────────
        if len(points) < self.MIN_POINTS:
            return True

        fx, fy, fw, fh = face_rect
        cx, cy = fx + fw / 2, fy + fh / 2

        # Build pixel polygon — points must already be in pixel coords
        polygon = np.array([[p[0], p[1]] for p in points], dtype=np.int32)
        result  = cv2.pointPolygonTest(polygon.reshape((-1, 1, 2)), (cx, cy), measureDist=False)
        return result >= 0   # ≥0 means on boundary or inside

    def is_face_inside_normalised(self, face_rect: tuple, frame_w: int, frame_h: int) -> bool:
        """
        Convenience: test face_rect (pixel) against the active normalised polygon.
        Converts zone points to pixel coords first.
        """
        pts = self.active_points
        if pts is None:
            return False   # No zone → no recognition

        pixel_pts = [(int(p["x"] * frame_w), int(p["y"] * frame_h)) for p in pts]
        return self.is_face_inside_zone(face_rect, pixel_pts)


zone_manager = ZoneManager()
