"""
bin/run_demo.py

Executable example script demonstrating how to clean the Charlotte traffic sample
dataset, train the Random Forest classifier, evaluate test metrics, and output
high-risk traffic predictions across Charlotte, NC.

Usage:
    python bin/run_demo.py
    python bin/run_demo.py --data charlotte_traffic.csv
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "hour",
    "minute",
    "is_weekend",
    "is_rush_hour",
    "road",
    "road_type",
    "road_class_frc",
    "temperature_f",
    "precipitation_in",
    "is_raining",
    "weather_code",
    "wind_speed_mph",
    "road_closure",
    "location",
]
BOOLEAN_COLS = ["is_weekend", "is_rush_hour", "is_raining", "road_closure"]
CATEGORICAL_COLS = ["road", "road_type", "road_class_frc", "location"]
NUMERIC_COLS = [col for col in FEATURE_COLS if col not in CATEGORICAL_COLS]


def clean_traffic_data(input_csv: str, output_csv: str) -> pd.DataFrame:
    """Loads raw Charlotte traffic CSV data, cleans columns/text, and extracts time features."""
    df = pd.read_csv(input_csv)
    df = df.drop_duplicates().copy()

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in text_columns:
        df[col] = df[col].str.strip()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.day_name()
        df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

    if output_csv:
        df.to_csv(output_csv, index=False)

    return df


def prepare_features_and_split(df: pd.DataFrame, train_ratio: float = 0.80) -> tuple:
    """Prepares features/labels and performs a chronological train/test split."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    X = df[FEATURE_COLS].copy()
    for col in BOOLEAN_COLS:
        X[col] = X[col].astype(int)

    # Binary target: 0 = No/low traffic, 1 = Traffic (medium or high)
    y = (df["traffic_level"] != "low").astype(int)

    df_sorted = df.sort_values("timestamp")
    split_index = int(len(df_sorted) * train_ratio)

    train_indices = df_sorted.index[:split_index]
    test_indices = df_sorted.index[split_index:]

    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

    return X_train, X_test, y_train, y_test, train_indices, test_indices


def build_and_train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Builds and fits the preprocessing and Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("numeric", StandardScaler(), NUMERIC_COLS),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model


def evaluate_traffic_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluates the trained model on the held-out test split."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No/low traffic", "Traffic"],
            zero_division=0,
        ),
        "confusion_df": pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=[0, 1]),
            index=["Actual no/low traffic", "Actual traffic"],
            columns=["Predicted no/low traffic", "Predicted traffic"],
        ),
    }


def predict_latest_locations(
    model: Pipeline, df: pd.DataFrame, threshold: float = 0.50, high_risk_only: bool = False
) -> pd.DataFrame:
    """Predicts congestion probabilities for the latest reading at each Charlotte location."""
    latest_by_location = (
        df.sort_values("timestamp")
        .groupby("location", as_index=False)
        .tail(1)
        .copy()
    )

    X_latest = latest_by_location[FEATURE_COLS].copy()
    for col in BOOLEAN_COLS:
        X_latest[col] = X_latest[col].astype(int)

    latest_by_location["traffic_probability"] = model.predict_proba(X_latest)[:, 1]
    latest_by_location["prediction"] = latest_by_location["traffic_probability"].apply(
        lambda prob: "Traffic likely" if prob >= threshold else "Low traffic likely"
    )

    predictions = latest_by_location.sort_values(by="traffic_probability", ascending=False)
    if high_risk_only:
        predictions = predictions[predictions["traffic_probability"] >= threshold].copy()

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run the Charlotte Traffic Congestion Prediction demo pipeline."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(REPO_ROOT / "charlotte_traffic.csv"),
        help="Path to the input traffic CSV file (default: charlotte_traffic.csv)",
    )
    args = parser.parse_args()

    print(f"[1/4] Cleaning dataset: {args.data}")
    df = clean_traffic_data(
        input_csv=args.data,
        output_csv=str(REPO_ROOT / "cleaned_charlotte_traffic.csv"),
    )
    print(f"      Loaded {len(df)} rows across {df['location'].nunique()} Charlotte locations.")

    print("[2/4] Preparing features and chronological 80/20 train/test split...")
    X_train, X_test, y_train, y_test, _, _ = prepare_features_and_split(df, train_ratio=0.80)
    print(f"      Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

    print("[3/4] Training balanced Random Forest pipeline...")
    model = build_and_train_model(X_train, y_train)

    print("[4/4] Evaluating model on held-out test set...\n")
    results = evaluate_traffic_model(model, X_test, y_test)
    print("=== Model Evaluation ===")
    print(f"Accuracy:          {results['accuracy']:.3f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.3f}\n")
    print("=== Classification Report ===")
    print(results["report"])
    print("=== Confusion Matrix ===")
    print(results["confusion_df"])

    print("\n=== High-Risk Charlotte Traffic Locations (>= 50% Probability) ===")
    high_risk = predict_latest_locations(model, df, threshold=0.50, high_risk_only=True)
    high_risk_display = high_risk[
        ["location", "road", "current_speed_mph", "free_flow_speed_mph", "traffic_probability", "prediction"]
    ].copy()
    high_risk_display["traffic_probability"] = high_risk_display["traffic_probability"].map("{:.1%}".format)
    print(high_risk_display.to_string(index=False))


if __name__ == "__main__":
    main()
