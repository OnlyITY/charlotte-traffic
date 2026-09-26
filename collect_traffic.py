# Charlotte traffic data collection
# Gets live traffic speeds for 25 Charlotte locations from the TomTom Traffic Flow API,
# adds weather and time info, labels traffic as low/medium/high, and saves it to a CSV.
#
# Run once:              python collect_traffic.py
# Run every 15 minutes:  python collect_traffic.py 15

import os
import sys
import csv
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

# API key is stored in .env so it doesn't get uploaded to GitHub
load_dotenv()
API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()

CSV_FILE = "charlotte_traffic.csv"

# Lower zoom = only major roads, so interstate points match the highway and not a side street
ZOOM = {"interstate": 6, "arterial": 9}

# name, road, road type, latitude, longitude
LOCATIONS = [
    ("I-77 at Arrowood Rd", "I-77", "interstate", 35.1380, -80.9020),
    ("I-77 at Tyvola Rd", "I-77", "interstate", 35.1670, -80.8840),
    ("I-77 at John Belk Fwy", "I-77", "interstate", 35.2155, -80.8620),
    ("I-77 at LaSalle St", "I-77", "interstate", 35.2580, -80.8380),
    ("I-77 at Sunset Rd", "I-77", "interstate", 35.3300, -80.8250),
    ("I-85 at Little Rock Rd", "I-85", "interstate", 35.2350, -80.9350),
    ("I-85 at Beatties Ford Rd", "I-85", "interstate", 35.2650, -80.8670),
    ("I-85 at Statesville Ave", "I-85", "interstate", 35.2680, -80.8450),
    ("I-85 at Sugar Creek Rd", "I-85", "interstate", 35.2620, -80.7850),
    ("I-85 at W.T. Harris Blvd", "I-85", "interstate", 35.3050, -80.7450),
    ("I-485 at I-77 South", "I-485", "interstate", 35.0900, -80.9170),
    ("I-485 at Johnston Rd", "I-485", "interstate", 35.0520, -80.8460),
    ("I-485 at Providence Rd", "I-485", "interstate", 35.0500, -80.7720),
    ("I-485 at Independence Blvd", "I-485", "interstate", 35.1280, -80.7000),
    ("I-485 at Albemarle Rd", "I-485", "interstate", 35.2000, -80.6800),
    ("I-277 Brookshire Fwy", "I-277", "interstate", 35.2370, -80.8430),
    ("I-277 John Belk Fwy", "I-277", "interstate", 35.2120, -80.8350),
    ("Uptown (Trade & Tryon)", "Tryon St", "arterial", 35.2271, -80.8431),
    ("Independence Blvd at Wendover", "US-74", "arterial", 35.1980, -80.7950),
    ("South Blvd at Woodlawn Rd", "South Blvd", "arterial", 35.1795, -80.8745),
    ("Providence Rd at Sharon Amity", "Providence Rd", "arterial", 35.1608, -80.7897),
    ("N Tryon St at University", "N Tryon St", "arterial", 35.3080, -80.7430),
    ("Billy Graham Pkwy at Airport", "Billy Graham Pkwy", "arterial", 35.2200, -80.9180),
    ("W.T. Harris Blvd at Albemarle", "W.T. Harris Blvd", "arterial", 35.2050, -80.7350),
    ("Wilkinson Blvd", "Wilkinson Blvd", "arterial", 35.2300, -80.9000),
]

HEADERS = [
    "timestamp", "date", "day_of_week", "hour", "minute", "is_weekend", "is_rush_hour",
    "location", "road", "road_type", "latitude", "longitude",
    "snapped_latitude", "snapped_longitude", "road_class_frc",
    "current_speed_mph", "free_flow_speed_mph",
    "current_travel_time_s", "free_flow_travel_time_s", "delay_s",
    "confidence", "road_closure", "speed_ratio",
    "temperature_f", "precipitation_in", "is_raining", "weather_code", "wind_speed_mph",
    "traffic_level",
]


def traffic_level(ratio):
    # ratio = current speed / free-flow speed (free-flow = speed with no traffic)
    if ratio > 0.8:
        return "low"
    elif ratio > 0.5:
        return "medium"
    else:
        return "high"


def get_weather():
    # Open-Meteo is free and doesn't need a key. One call per run for all of Charlotte.
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 35.2271,
        "longitude": -80.8431,
        "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }
    try:
        w = requests.get(url, params=params, timeout=15).json()["current"]
        rain = w["precipitation"] or 0
        return {
            "temperature_f": w["temperature_2m"],
            "precipitation_in": rain,
            "is_raining": rain > 0,
            "weather_code": w["weather_code"],
            "wind_speed_mph": w["wind_speed_10m"],
        }
    except Exception:
        # if weather fails, still save the traffic data with blank weather
        print("Couldn't get weather this run")
        return {"temperature_f": "", "precipitation_in": "", "is_raining": "",
                "weather_code": "", "wind_speed_mph": ""}


def collect():
    now = datetime.now()
    weather = get_weather()
    weekend = now.weekday() >= 5
    rush_hour = not weekend and now.hour in (7, 8, 16, 17)  # 7-9am and 4-6pm

    rows = []
    for name, road, road_type, lat, lon in LOCATIONS:
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{ZOOM[road_type]}/json"
        params = {"key": API_KEY, "point": f"{lat},{lon}", "unit": "mph"}
        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            print(f"{name}: error {response.status_code}")
            continue

        data = response.json()["flowSegmentData"]

        # print one raw response to show what the API sends back
        if not rows:
            print("Sample API response:", {k: v for k, v in data.items() if k != "coordinates"})
            print()

        # middle point of the road segment TomTom matched
        points = data.get("coordinates", {}).get("coordinate", [])
        mid = points[len(points) // 2] if points else {}

        ratio = round(data["currentSpeed"] / (data["freeFlowSpeed"] or 1), 2)

        rows.append({
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "hour": now.hour,
            "minute": now.minute,
            "is_weekend": weekend,
            "is_rush_hour": rush_hour,
            "location": name,
            "road": road,
            "road_type": road_type,
            "latitude": lat,
            "longitude": lon,
            "snapped_latitude": mid.get("latitude", ""),
            "snapped_longitude": mid.get("longitude", ""),
            "road_class_frc": data["frc"],  # FRC0 = freeway, higher = smaller road
            "current_speed_mph": data["currentSpeed"],
            "free_flow_speed_mph": data["freeFlowSpeed"],
            "current_travel_time_s": data["currentTravelTime"],
            "free_flow_travel_time_s": data["freeFlowTravelTime"],
            "delay_s": data["currentTravelTime"] - data["freeFlowTravelTime"],
            "confidence": data["confidence"],
            "road_closure": data["roadClosure"],
            "speed_ratio": ratio,
            **weather,
            "traffic_level": traffic_level(ratio),
        })
        time.sleep(0.25)  # small pause between API calls

    # add header only if the file is new, then append the rows
    new_file = not os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"{r['location']:<32} {r['road_class_frc']:<5} {r['current_speed_mph']:>3} / "
              f"{r['free_flow_speed_mph']:>3} mph  {r['traffic_level']}")

    counts = {lvl: sum(r["traffic_level"] == lvl for r in rows) for lvl in ("low", "medium", "high")}
    print(f"\nWeather: {weather['temperature_f']}F")
    print(f"Saved {len(rows)} rows (low {counts['low']}, medium {counts['medium']}, high {counts['high']})")


if __name__ == "__main__":
    if not API_KEY:
        print("Missing API key - add TOMTOM_API_KEY to .env")
        sys.exit()

    # no number = run once, a number = repeat every that many minutes
    if len(sys.argv) > 1:
        minutes = float(sys.argv[1])
        while True:
            print(f"\n--- Run at {datetime.now():%Y-%m-%d %H:%M:%S} ---")
            collect()
            print(f"Next run in {minutes:g} minutes (Ctrl+C to stop)")
            time.sleep(minutes * 60)
    else:
        collect()