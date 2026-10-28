"""Baseline classification workflow for the synthetic insurance claims dataset.

Run from the project root:
    python src/train_baseline.py

The script compares baseline classifiers using cross-validation on the training
set, then evaluates the selected candidate once on a held-out test set.
This is a learning demo, not a production insurance decision system.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET = "manual_review_required"
ID_COLUMN = "claim_id"
DATE_COLUMN = "incident_date"
RANDOM_STATE = 42


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "data" / "insurance_claims_practice_dataset.csv",
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs",
        help="Directory for metrics and the saved model.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Place the CSV in the data/ folder or pass --data PATH."
        )

    df = pd.read_csv(args.data)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing from the dataset.")
    if df[TARGET].isna().any():
        raise ValueError(f"Target column '{TARGET}' contains missing labels.")

    # Clean whitespace in target labels; keep the intended Yes/No labels explicit.
    df[TARGET] = df[TARGET].astype(str).str.strip()
    observed_labels = set(df[TARGET].unique())
    expected_labels = {"Yes", "No"}
    if observed_labels != expected_labels:
        raise ValueError(
            f"Expected target labels {sorted(expected_labels)}, found {sorted(observed_labels)}."
        )

    print("=== 1. Data validation ===")
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print("Missing values by column:")
    missing = df.isna().sum()
    print(missing[missing.gt(0)].sort_values(ascending=False).to_string() or "None")
    print("\nTarget counts:")
    print(df[TARGET].value_counts().to_string())
    print("\nTarget percentages:")
    print((df[TARGET].value_counts(normalize=True).mul(100).round(1)).astype(str).add("%").to_string())

    # Baseline intentionally excludes unique IDs and raw dates. Date parts can be
    # explored later if there is a valid business reason to use time information.
    exclusions = [TARGET]
    for column in (ID_COLUMN, DATE_COLUMN):
        if column in df.columns:
            exclusions.append(column)
    X = df.drop(columns=exclusions)
    y = df[TARGET]

    if X.shape[1] == 0:
        raise ValueError("No predictor columns remain after excluding ID/date/target fields.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print("\n=== 2. Train/test split ===")
    print(f"Training rows: {len(X_train)}; test rows: {len(X_test)}")
    print("Training target counts:", y_train.value_counts().to_dict())
    print("Test target counts:", y_test.value_counts().to_dict())

    # Each pipeline learns imputation/encoding/scaling from training folds only.
    candidate_models = {
        "Dummy baseline": Pipeline(
            [
                ("prep", build_preprocessor(X_train, scale_numeric=False)),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("prep", build_preprocessor(X_train, scale_numeric=True)),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("prep", build_preprocessor(X_train, scale_numeric=False)),
                ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", build_preprocessor(X_train, scale_numeric=False)),
                ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
    }

    scoring = {
        "f1_yes": make_scorer(f1_score, pos_label="Yes", zero_division=0),
        "precision_yes": make_scorer(precision_score, pos_label="Yes", zero_division=0),
        "recall_yes": make_scorer(recall_score, pos_label="Yes", zero_division=0),
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n=== 3. Cross-validation on training data ===")
    rows = []
    for name, model in candidate_models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            row[f"cv_{metric}"] = scores[f"test_{metric}"].mean()
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values("cv_f1_yes", ascending=False)
    print(comparison.round(3).to_string(index=False))

    # Model choice is made from training-fold CV, not by repeatedly looking at test results.
    best_name = str(comparison.iloc[0]["model"])
    best_model = candidate_models[best_name]
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    print(f"\n=== 4. Held-out test evaluation: {best_name} ===")
    metrics = {
        "selected_model_by_cv_f1_yes": best_name,
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "test_precision_yes": precision_score(y_test, predictions, pos_label="Yes", zero_division=0),
        "test_recall_yes": recall_score(y_test, predictions, pos_label="Yes", zero_division=0),
        "test_f1_yes": f1_score(y_test, predictions, pos_label="Yes", zero_division=0),
    }
    for key, value in metrics.items():
        print(f"{key}: {value if isinstance(value, str) else f'{value:.3f}'}")

    matrix = confusion_matrix(y_test, predictions, labels=["No", "Yes"])
    print("\nConfusion matrix (rows=actual [No, Yes], columns=predicted [No, Yes]):")
    print(matrix)
    report = classification_report(y_test, predictions, labels=["No", "Yes"], zero_division=0)
    print("\nClassification report:")
    print(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "cross_validation_comparison.csv", index=False)
    joblib.dump(best_model, args.output_dir / "insurance_claims_baseline_model.joblib")
    prediction_output = pd.DataFrame(
        {
            "actual_manual_review_required": y_test,
            "predicted_manual_review_required": predictions,
        },
        index=y_test.index,
    ).sort_index()
    if ID_COLUMN in df.columns:
        prediction_output.insert(0, ID_COLUMN, df.loc[prediction_output.index, ID_COLUMN])
    prediction_output.to_csv(args.output_dir / "test_predictions.csv", index=False)

    metrics_text = ["Insurance claims baseline results", "", f"Dataset: {args.data}", ""]
    metrics_text.extend(f"{key}: {value if isinstance(value, str) else f'{value:.4f}'}" for key, value in metrics.items())
    metrics_text.extend(["", "Confusion matrix (actual rows [No, Yes] × predicted columns [No, Yes]):", str(matrix), "", "Classification report:", report])
    (args.output_dir / "test_metrics.txt").write_text("\n".join(metrics_text), encoding="utf-8")

    print(f"\nSaved outputs to: {args.output_dir}")
    print("Reminder: this is a synthetic learning dataset, not suitable for real insurance decisions.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
