"""
bin/run_demo.py

Executable example script demonstrating how to clean the Charlotte traffic sample
dataset, train the Random Forest classifier, evaluate test metrics, and output
high-risk traffic predictions across Charlotte, NC.

Usage:
    python bin/run_demo.py --data charlotte_traffic.csv
"""

import argparse
import sys
from pathlib import Path

# Add repository root to sys.path so traffic_pipeline can be imported
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from traffic_pipeline import (
    build_and_train_model,
    clean_traffic_data,
    evaluate_traffic_model,
    predict_latest_locations,
    prepare_features_and_split,
)


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