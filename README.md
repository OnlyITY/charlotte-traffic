# Charlotte Traffic Congestion Prediction

A traffic prediction tool for Charlotte, NC that predicts whether traffic at a location will be low, medium, or high.

This repo is the data collection part of the project. `collect_traffic.py` pulls live traffic data from the TomTom Traffic Flow API and saves it to `charlotte_traffic.csv`, which is used for the S3 storage step and the ML model.

## How it works

1. The script calls the TomTom Traffic Flow API for 25 Charlotte locations on I-77, I-85, I-485, I-277, and major roads like Independence Blvd, South Blvd, and Providence Rd.
2. It gets the current Charlotte weather from the Open-Meteo API (free, no key needed).
3. Each reading is labeled by comparing the current speed to the free-flow speed (the speed with no traffic):
   - above 0.8 = low
   - 0.5 to 0.8 = medium
   - 0.5 or below = high
4. Each run adds 25 rows (one per location) to `charlotte_traffic.csv`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your own TomTom API key (free at developer.tomtom.com):

```
TOMTOM_API_KEY=your_key_here
```

`.env` is in `.gitignore`, so the key doesn't get uploaded to GitHub.

## Running

```bash
python collect_traffic.py        # one snapshot (25 rows)
python collect_traffic.py 15     # a snapshot every 15 minutes until Ctrl+C
```

To keep a Mac from sleeping during long collection: `caffeinate -i python collect_traffic.py 15`

25 locations every 15 minutes is about 2,400 requests per day, which stays under TomTom's free limit.

## CSV columns

| Column | Meaning |
| --- | --- |
| timestamp, date, day_of_week, hour, minute | When the reading was taken |
| is_weekend, is_rush_hour | Weekend flag; rush hour = weekdays 7-9 AM and 4-6 PM |
| location, road, road_type | Where the reading was taken (interstate or arterial) |
| latitude, longitude | The point sent to the API |
| snapped_latitude, snapped_longitude | The road segment TomTom matched |
| road_class_frc | TomTom road class (FRC0 = freeway, higher = smaller road) |
| current_speed_mph, free_flow_speed_mph | Current speed vs. speed with no traffic |
| current_travel_time_s, free_flow_travel_time_s, delay_s | Travel time across the segment and extra delay in seconds |
| confidence | TomTom data quality, 0 to 1 |
| road_closure | Whether the road is closed |
| speed_ratio | current speed / free-flow speed |
| temperature_f, precipitation_in, is_raining, weather_code, wind_speed_mph | Charlotte weather at the time of the reading |
| traffic_level | Label for the model: low, medium, or high |

## Notes

- The current CSV has 100 rows collected on Saturday, Sep 26, 2026 between about 12:15 and 1:05 PM (78 low, 14 medium, 8 high). Weekday rush hour data would add more medium and high examples.
- The labels are imbalanced (mostly low), so the model should use a stratified train/test split and class weights.
- `archive/` has earlier work: the NCDOT incidents API attempt (NCDOT retired it in May 2026), the first 5-location CSV, and a Saturday morning test run of the 25-location script.

## License

MIT License