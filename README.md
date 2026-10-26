# Charlotte Traffic Congestion Prediction

**Course:** DTSC 3601: Data Science Studio 3: Predictive Analysis & Their Impacts  
**Group Name:** 2027  
**Team Members:** Abhiram Parimi, Omar Osman, Justin Yip

A traffic prediction tool for Charlotte, NC that predicts whether traffic at a location will be low, medium, or high.

This repo includes the data collection (`collect_traffic.py`), data preparation (`data preparation.ipynb`), and machine learning (`Charlotte_Traffic_Prediction.ipynb`) parts of the project. `collect_traffic.py` pulls live traffic data from the TomTom Traffic Flow API and saves it to `charlotte_traffic.csv`, which is used for the S3 storage step and the ML model.

## How it works

1. The script calls the TomTom Traffic Flow API for 25 Charlotte locations on I-77, I-85, I-485, I-277, and major roads like Independence Blvd, South Blvd, and Providence Rd.
2. It gets the current Charlotte weather from the Open-Meteo API (free, no key needed).
3. Each reading is labeled by comparing the current speed to the free-flow speed (the speed with no traffic):
   - above 0.8 = low
   - 0.5 to 0.8 = medium
   - 0.5 or below = high
4. Each run adds 25 rows (one per location) to `charlotte_traffic.csv`.
5. `data preparation.ipynb` cleans and validates `charlotte_traffic.csv` and saves `cleaned_charlotte_traffic.csv` for AWS S3 and modeling.
6. `Charlotte_Traffic_Prediction.ipynb` (and [`bin/run_demo.py`](./bin/run_demo.py)) trains a class-balanced Random Forest classifier to predict traffic congestion across all 25 Charlotte locations.

---

## How to Install & Setup

### Required Libraries
* `requests==2.34.2` — Queries the TomTom Traffic Flow and Open-Meteo APIs
* `python-dotenv==1.2.3` — Loads the `TOMTOM_API_KEY` from `.env`
* `pandas` — Cleans, formats, and processes CSV traffic data
* `scikit-learn` — Builds the preprocessing pipeline (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`), trains the `RandomForestClassifier`, and calculates evaluation metrics
* `matplotlib` — Plots traffic level distributions and hourly speed trends
* `certifi==2026.7.22`, `charset-normalizer==3.5.1`, `idna==3.20`, `urllib3==2.8.0` — Supporting HTTP dependencies

### Code to Install

```bash
git clone [https://github.com/abhiramparimi05-bot/charlotte-traffic.git](https://github.com/abhiramparimi05-bot/charlotte-traffic.git)
cd charlotte-traffic

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pandas scikit-learn matplotlib
```

Create a `.env` file with your own TomTom API key (free at [developer.tomtom.com](https://developer.tomtom.com)):

```
TOMTOM_API_KEY=your_key_here
```

`.env` is in `.gitignore`, so the key doesn't get uploaded to GitHub.

---

## Running Data Collection

```bash
python collect_traffic.py        # one snapshot (25 rows)
python collect_traffic.py 15     # a snapshot every 15 minutes until Ctrl+C
```

To keep a Mac from sleeping during long collection: `caffeinate -i python collect_traffic.py 15`

25 locations every 15 minutes is about 2,400 requests per day, which stays under TomTom's free limit.

---

## Get Started: Quick Training Demo on a Small Dataset

You can quickly train and test our Random Forest model on the 100-row sample dataset (`cleaned_Charlotte_traffic.csv`) included in the repository without needing an API key.

### 1. Run via the `bin/` Example Script
Run our standalone demo script located in the [`bin/`](./bin/) directory:

```bash
python bin/run_demo.py
```

### 2. Run Quick Demo in Python
```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

# 1. Load the small 100-row dataset
df = pd.read_csv("cleaned_Charlotte_traffic.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 2. Select features and binary target (0 = No/low traffic, 1 = Traffic)
feature_cols = [
    "hour", "minute", "is_weekend", "is_rush_hour", "road", "road_type",
    "road_class_frc", "temperature_f", "precipitation_in", "is_raining",
    "weather_code", "wind_speed_mph", "road_closure", "location"
]
boolean_cols = ["is_weekend", "is_rush_hour", "is_raining", "road_closure"]
categorical_cols = ["road", "road_type", "road_class_frc", "location"]
numeric_cols = [col for col in feature_cols if col not in categorical_cols]

X = df[feature_cols].copy()
for col in boolean_cols:
    X[col] = X[col].astype(int)
y = (df["traffic_level"] != "low").astype(int)

# 3. Chronological 80/20 train/test split (80 train rows, 20 test rows)
df_sorted = df.sort_values("timestamp")
split_index = int(len(df_sorted) * 0.80)
train_indices, test_indices = df_sorted.index[:split_index], df_sorted.index[split_index:]
X_train, X_test = X.loc[train_indices], X.loc[test_indices]
y_train, y_test = y.loc[train_indices], y.loc[test_indices]

# 4. Build and train the Random Forest pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numeric", StandardScaler(), numeric_cols),
    ]
)
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("random_forest", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ]
)
model.fit(X_train, y_train)

# 5. Evaluate model performance
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=["No/low traffic", "Traffic"]))
```

---

## Documentation: Major Functions & How They Work

Below is detailed documentation of all major functions and pipeline workflows in this repository (modeled after [PyTorch Hub](https://pytorch.org/hub/)):

### 1. `traffic_level(ratio)` (`collect_traffic.py`)
Labels each traffic reading as `"low"`, `"medium"`, or `"high"` congestion by comparing the current road speed to the free-flow speed.
* **Parameters:**
  * `ratio` (*float*): `current_speed_mph / free_flow_speed_mph`.
* **Returns:**
  * *str*: `"low"` if `ratio > 0.8`, `"medium"` if `0.5 < ratio <= 0.8`, or `"high"` if `ratio <= 0.5`.

### 2. `get_weather()` (`collect_traffic.py`)
Fetches real-time weather conditions for Charlotte, NC (`latitude=35.2271, longitude=-80.8431`) from the Open-Meteo API (`https://api.open-meteo.com/v1/forecast`).
* **Parameters:** None.
* **Returns:**
  * *dict*: Contains `temperature_f` (*float*), `precipitation_in` (*float*), `is_raining` (*bool*), `weather_code` (*int*), and `wind_speed_mph` (*float*). If the API call fails, returns blank strings so traffic collection still succeeds.

### 3. `collect()` (`collect_traffic.py`)
Loops through all 25 Charlotte locations (`LOCATIONS`), calls the TomTom Traffic Flow Segment API (`flowSegmentData/absolute/{zoom}/json` with `zoom=6` for interstates and `zoom=9` for arterials), pairs the response with `get_weather()`, computes speed ratios and travel delays, and appends 25 rows to `charlotte_traffic.csv`.
* **Parameters:** None (uses `TOMTOM_API_KEY` loaded from `.env`).
* **Returns:** None (writes directly to `charlotte_traffic.csv` and prints a summary table to the terminal).

### 4. Data Preparation Pipeline (`data preparation.ipynb`)
Prepares raw API records for AWS S3 storage and machine learning:
* **Data Validation & Deduplication:** Checks `df.info()`, verifies missing values (`df.isnull().sum()`), and drops duplicate rows (`df.drop_duplicates()`).
* **Column & String Cleaning:** Standardizes column names (`str.strip().str.lower().str.replace(" ", "_")`) and trims whitespace from text columns.
* **Datetime Feature Extraction:** Converts `timestamp` using `pd.to_datetime()` and extracts `hour`, `day_of_week`, and `is_weekend`.
* **Export:** Saves cleaned records to `cleaned_charlotte_traffic.csv`.

### 5. Machine Learning Pipeline (`Charlotte_Traffic_Prediction.ipynb` & [`bin/run_demo.py`](./bin/run_demo.py))
Preprocesses features, trains the classifier, and forecasts congestion probabilities by location:
* **`preprocessor` (`ColumnTransformer`):** One-hot encodes categorical columns (`road`, `road_type`, `road_class_frc`, `location`) using `OneHotEncoder(handle_unknown="ignore")` and standardizes numeric/boolean columns (`hour`, `minute`, `is_weekend`, `is_rush_hour`, `temperature_f`, `precipitation_in`, `is_raining`, `weather_code`, `wind_speed_mph`, `road_closure`) using `StandardScaler()`.
* **`model` (`Pipeline`):** Combines `preprocessor` with `RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)`. Achieves `0.950` accuracy and `0.967` balanced accuracy on the test split.
* **High-Risk Location Forecasting:** Selects the most recent reading per location (`df.sort_values("timestamp").groupby("location").tail(1)`), calculates `model.predict_proba(X_latest)[:, 1]`, and labels locations with `>= 50%` probability as `"Traffic likely"`.

---

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

---

## Notes

- The current CSV has 100 rows collected on Saturday, Sep 26, 2026 between about 12:15 and 1:05 PM (78 low, 14 medium, 8 high). Weekday rush hour data would add more medium and high examples.
- The labels are imbalanced (mostly low), so the model should use a stratified train/test split and class weights.
- `archive/` has earlier work: the NCDOT incidents API attempt (NCDOT retired it in May 2026), the first 5-location CSV, and a Saturday morning test run of the 25-location script.
- [`bin/run_demo.py`](./bin/run_demo.py) provides an executable command-line example of how to run the model training and prediction pipeline.

---

## Contact

Who to contact with questions:
* **Abhiram Parimi** — aparimi@charlotte.edu ([GitHub](https://github.com/abhiramparimi05-bot))
* **Omar Osman** — oosman2@charlotte.edu
* **Justin Yip** — iyip@charlotte.edu

---

## License

This project is licensed under the [MIT License](./LICENSE).
