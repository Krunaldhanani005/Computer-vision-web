from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["fr_surveillance_db"]

# Drop old collections
for col in ["faces", "zones", "logs", "known_reports", "unknown_reports", "blacklist_reports", "attendance_reports"]:
    db[col].drop()
    print(f"Dropped {col}")

print("FR database cleared successfully.")
