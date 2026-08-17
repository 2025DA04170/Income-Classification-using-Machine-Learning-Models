"""
train_models.py
----------------
Trains 5 classification models on the UCI Adult / Census Income dataset:
Logistic Regression, Decision Tree, K-Nearest Neighbour, Gaussian Naive Bayes,
and Random Forest (Ensemble).

For each model this script computes: Accuracy, AUC, Precision, Recall, F1, MCC.

Outputs:
- ../adult_cleaned.csv          (cleaned dataset)
- ../test_data.csv              (held-out test split, used by the Streamlit app)
- logistic_regression_model.pkl
- decision_tree_model.pkl
- knn_model.pkl
- naive_bayes_model.pkl
- random_forest_model.pkl
- preprocessor.pkl
- metrics.csv                   (comparison table used in the README)

Run from inside the model/ folder:
    python train_models.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "..", "adult_income_raw.csv")
CLEANED_PATH = os.path.join(HERE, "..", "adult_cleaned.csv")
TEST_DATA_PATH = os.path.join(HERE, "..", "test_data.csv")
METRICS_PATH = os.path.join(HERE, "..", "model", "metrics.csv")

TARGET = "income"

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Replace literal "?" (with or without surrounding whitespace) with NaN
    df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)

    # Drop rows with a missing target
    df = df.dropna(subset=[TARGET])

    # Normalise target labels (strip stray periods some UCI exports contain, e.g. "<=50K.")
    df[TARGET] = df[TARGET].astype(str).str.strip().str.replace(".", "", regex=False)

    # Encode target: <=50K -> 0, >50K -> 1
    df[TARGET] = df[TARGET].map({"<=50K": 0, ">50K": 1})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest (Ensemble)",
}


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    print("Loading and cleaning dataset...")
    df = load_and_clean(RAW_PATH)
    df.to_csv(CLEANED_PATH, index=False)
    print(f"Cleaned dataset saved to {CLEANED_PATH} ({len(df)} rows)")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Save a test_data.csv (features + true label) for the Streamlit app to demo on.
    # Free-tier friendly: capped at 1000 rows.
    test_export = X_test.copy()
    test_export[TARGET] = y_test.map({0: "<=50K", 1: ">50K"}).values
    test_export.head(1000).to_csv(TEST_DATA_PATH, index=False)
    print(f"Test data saved to {TEST_DATA_PATH} ({len(test_export.head(1000))} rows)")

    print("Fitting preprocessor...")
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    joblib.dump(preprocessor, os.path.join(HERE, "preprocessor.pkl"), compress=3)

    results = []
    models = get_models()
    for key, model in models.items():
        print(f"Training {DISPLAY_NAMES[key]}...")
        model.fit(X_train_t, y_train)
        metrics = evaluate(model, X_test_t, y_test)
        metrics["Model"] = DISPLAY_NAMES[key]
        results.append(metrics)

        model_path = os.path.join(HERE, f"{key}_model.pkl")
        joblib.dump(model, model_path, compress=3)
        print(f"  Saved -> {model_path}")
        print(f"  {metrics}")

    metrics_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    metrics_df.to_csv(METRICS_PATH, index=False)
    print("\nFinal comparison table:")
    print(metrics_df.to_string(index=False))
    print(f"\nMetrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
