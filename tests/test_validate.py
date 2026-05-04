from services.zones.zone_manager import zone_manager
import json

zone = {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}
print(zone_manager.validate_coordinates(zone))
