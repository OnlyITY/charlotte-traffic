import os
import csv
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = (os.getenv("TOMTOM_API_KEY") or "").strip()
print(f"Key loaded: {len(API_KEY)} characters (starts with {API_KEY[:4]})")

# Charlotte locations
points = {
    "Uptown Charlotte": "35.2271,-80.8431",
    "I-77 South": "35.1900,-80.8750",
    "I-85 North": "35.2900,-80.7800",
    "I-485 South": "35.0600,-80.8200",
    "Independence Blvd": "35.1950,-80.7700",
}

CSV_FILE = "charlotte_traffic.csv"
HEADERS = ["timestamp", "location", "latitude", "longitude",
           "current_speed_kmh", "free_flow_speed_kmh",
           "current_travel_time_s", "free_flow_travel_time_s",
           "confidence", "road_closure", "speed_ratio", "traffic_level"]

# Connect to TomTom Traffic Flow API
rows = []
for name, pt in points.items():
    url = ("https://api.tomtom.com/traffic/services/4/flowSegmentData/"
           f"absolute/10/json?point={pt}&key={API_KEY}")
    response = requests.get(url, timeout=30)
    print(f"\n{name}: status {response.status_code}")

    if response.status_code != 200:
        print(response.text)
        continue

    data = response.json()["flowSegmentData"]
    print(json.dumps({k: v for k, v in data.items() if k != "coordinates"}, indent=2))

    ratio = round(data["currentSpeed"] / data["freeFlowSpeed"], 2)
    level = "low" if ratio > 0.8 else "medium" if ratio > 0.5 else "high"
    lat, lon = pt.split(",")

    rows.append([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, lat, lon,
        data["currentSpeed"], data["freeFlowSpeed"],
        data["currentTravelTime"], data["freeFlowTravelTime"],
        data["confidence"], data["roadClosure"], ratio, level
    ])

# Save to CSV (header only the first time)
if rows:
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADERS)
        writer.writerows(rows)

    print("\n=== Summary ===")
    for r in rows:
        print(f"{r[1]:<20} {r[4]:>3} / {r[5]:>3} km/h  -> {r[11]}")
    print(f"\nSaved {len(rows)} rows to {CSV_FILE}")
else:
    print("\nNo data saved.")