import csv
import json
import requests
from datetime import datetime

# 1. Connect to the NCDOT traffic incidents API (no key needed)
URL = "https://eapps.ncdot.gov/services/traffic-prod/v1/incidents"
response = requests.get(URL, timeout=60)
print("Status code:", response.status_code)
response.raise_for_status()

data = response.json()

# Handle the response whether it's a list or wrapped in an object
if isinstance(data, list):
    incidents = data
elif isinstance(data, dict):
    incidents = next((v for v in data.values() if isinstance(v, list)), [data])
else:
    incidents = []

print(f"Total incidents returned by API: {len(incidents)}")

# 2. Mark which incidents are in the Charlotte area
def is_charlotte(incident):
    text = json.dumps(incident).lower()
    return "mecklenburg" in text or "charlotte" in text

charlotte_count = sum(1 for i in incidents if is_charlotte(i))
print(f"Incidents in Charlotte / Mecklenburg: {charlotte_count}")

# Show a sample of the raw API data in the VS Code terminal
if incidents:
    sample = next((i for i in incidents if is_charlotte(i)), incidents[0])
    print("\n--- Sample incident from API ---")
    print(json.dumps(sample, indent=2)[:1500])

# 3. Save all incidents to CSV
collected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
rows = []
for inc in incidents:
    flat = {"collected_at": collected_at, "is_charlotte": is_charlotte(inc)}
    for k, v in inc.items():
        flat[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    rows.append(flat)

if rows:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open("nc_traffic_incidents.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} incidents to nc_traffic_incidents.csv")
else:
    print("\nAPI returned no incidents.")