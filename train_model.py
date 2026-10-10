
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"


def main():
    print("Loading dataset...")

    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print("Missing columns:", missing)
        print("Check that you downloaded the original UCI CSV.")
        return

    df = df[FEATURES + [TARGET]].dropna().copy()
    df[TARGET] = df[TARGET].astype(int)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessing = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
        ("numeric", "passthrough", FEATURES[1:]),
    ])

    model = Pipeline([
        ("preprocess", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("\n===== MODEL EVALUATION =====")
    print(f"Dataset rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Failure precision: {precision:.4f}")
    print(f"Failure recall: {recall:.4f}")
    print(f"Failure F1-score: {f1:.4f}")
    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Failure", "Failure"],
        zero_division=0,
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_DIR / "predictive_maintenance_model.joblib",
    )

    metrics = {
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "accuracy": float(accuracy),
        "failure_precision": float(precision),
        "failure_recall": float(recall),
        "failure_f1": float(f1),
        "confusion_matrix": matrix.tolist(),
    }

    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nModel training completed!")
    print("Saved model and metrics inside the models folder.")


if __name__ == "__main__":
    main()
