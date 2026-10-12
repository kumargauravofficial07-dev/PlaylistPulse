from fastapi import APIRouter
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


router = APIRouter(
    prefix="/api",
    tags=["Master Results"]
)


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "spotify_processed.csv"
)

CLUSTERED_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clustered_tracks.csv"
)

MODEL_DIR = BASE_DIR / "models"


FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]


def load_dataset():
    return pd.read_csv(DATA_FILE)


def safe_float(value):
    if pd.isna(value):
        return 0.0
    return float(value)


def load_model(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    return joblib.load(path)


# ---------------------------------------------------------
# Reproduce the EXACT same train/test splits used when the
# models were originally trained. This is required so that
# metrics are calculated on held-out test data only, never
# on rows the model already saw during training.
# ---------------------------------------------------------

def get_classification_split(df):
    """
    Matches the split used in classification.py, tree_models.py,
    and boosting.py: test_size=0.2, random_state=42, stratify=y
    on the 'popular' target.
    """
    X = df[FEATURES]
    y = df["popular"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_test, y_test


def get_regression_split(df):
    """
    Matches the split used in regression.py: test_size=0.2,
    random_state=42 (no stratify) on the 'popularity' target.
    """
    X = df[FEATURES]
    y = df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_test, y_test


# ---------------------------------------------------------
# Dataset / EDA
# ---------------------------------------------------------

def get_dataset_results(df):
    class_counts = (
        df["popular"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isnull().sum().sum()),
        "popularity": {
            "mean": safe_float(df["popularity"].mean()),
            "median": safe_float(df["popularity"].median()),
            "min": safe_float(df["popularity"].min()),
            "max": safe_float(df["popularity"].max())
        },
        "class_distribution": {
            "not_popular": int(class_counts.get(0, 0)),
            "popular": int(class_counts.get(1, 0))
        }
    }


def get_eda_results(df):
    audio_statistics = {}

    for feature in FEATURES:
        audio_statistics[feature] = {
            "mean": safe_float(df[feature].mean()),
            "min": safe_float(df[feature].min()),
            "max": safe_float(df[feature].max())
        }

    genre_counts = (
        df["track_genre"]
        .value_counts()
        .head(15)
    )

    popularity_distribution = (
        df["popularity"]
        .value_counts()
        .sort_index()
    )

    correlations = (
        df[FEATURES + ["popularity"]]
        .corr()["popularity"]
        .drop("popularity")
        .sort_values(ascending=False)
    )

    return {
        "audio_statistics": audio_statistics,
        "popularity_distribution": [
            {"popularity": int(index), "count": int(value)}
            for index, value in popularity_distribution.items()
        ],
        "top_genres": [
            {"genre": str(index), "count": int(value)}
            for index, value in genre_counts.items()
        ],
        "popularity_correlations": [
            {"feature": str(index), "correlation": safe_float(value)}
            for index, value in correlations.items()
        ]
    }


# ---------------------------------------------------------
# Regression (evaluated on held-out test split only)
# All 4 models are loaded and compared. Linear Regression is
# always reported as the primary/interpretable model.
# ---------------------------------------------------------

def calculate_regression_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    actual = y_test.values

    residuals = actual - predictions

    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }, predictions


def get_regression_results(df):
    X_test, y_test = get_regression_split(df)

    model_files = {
        "Linear Regression": "regression_linear.pkl",
        "Ridge Regression": "regression_ridge.pkl",
        "Lasso Regression": "regression_lasso.pkl",
        "Elastic Net": "regression_elasticnet.pkl"
    }

    comparison = {}
    primary_predictions = None

    for name, filename in model_files.items():
        model = load_model(filename)

        if model is None:
            comparison[name] = {"status": "model_not_found"}
            continue

        metrics, predictions = calculate_regression_metrics(model, X_test, y_test)
        comparison[name] = metrics

        if name == "Linear Regression":
            primary_predictions = predictions

    points = []
    if primary_predictions is not None:
        actual = y_test.values
        for actual_value, predicted_value in zip(actual[:500], primary_predictions[:500]):
            points.append({
                "actual": round(float(actual_value), 2),
                "predicted": round(float(predicted_value), 2)
            })

    return {
        "primary_model": "Linear Regression",
        "note": "Linear Regression is presented as the primary/interpretable model. All four models are shown below for comparison; their scores are close because none captures much variance in popularity from audio features alone.",
        "comparison": comparison,
        "metrics": comparison.get("Linear Regression", {}),
        "actual_vs_predicted": points
    }


# ---------------------------------------------------------
# Classification (evaluated on held-out test split only)
# ---------------------------------------------------------

def calculate_classification_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    matrix = confusion_matrix(y_test, predictions)

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "confusion_matrix": matrix.tolist()
    }


def get_classification_results(df):
    model = load_model("classification_model.pkl")

    if model is None:
        return {"status": "model_not_found"}

    X_test, y_test = get_classification_split(df)

    metrics = calculate_classification_metrics(model, X_test, y_test)

    return {
        "model": "Logistic Regression",
        "target": "popular",
        "metrics": metrics
    }


# ---------------------------------------------------------
# Trees (evaluated on held-out test split only)
# ---------------------------------------------------------

def get_tree_results(df):
    results = {}

    X_test, y_test = get_classification_split(df)

    models = {
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl"
    }

    for name, filename in models.items():
        model = load_model(filename)
        if model is None:
            continue

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4)
        }

        if name == "Random Forest":
            if hasattr(model, "oob_score_"):
                results[name]["oob_score"] = round(float(model.oob_score_), 4)

            if hasattr(model, "feature_importances_"):
                importance = sorted(
                    zip(FEATURES, model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True
                )
                results[name]["feature_importance"] = [
                    {"feature": feature, "importance": round(float(value), 6)}
                    for feature, value in importance
                ]

    return results


# ---------------------------------------------------------
# Boosting (evaluated on held-out test split only)
# ---------------------------------------------------------

def get_boosting_results(df):
    results = {}

    X_test, y_test = get_classification_split(df)

    models = {
        "AdaBoost": "adaboost_model.pkl",
        "Gradient Boosting": "gradient_boosting_model.pkl",
        "XGBoost": "xgboost_model.pkl"
    }

    for name, filename in models.items():
        model = load_model(filename)
        if model is None:
            continue

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4)
        }

    return results


# ---------------------------------------------------------
# Clustering
# ---------------------------------------------------------

def get_clustering_results():
    if not CLUSTERED_FILE.exists():
        return {"status": "clustered_dataset_not_found"}

    df = pd.read_csv(CLUSTERED_FILE)

    distribution = (
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    return {
        "model": "MiniBatch K-Means",
        "clusters": int(df["cluster"].nunique()),
        "cluster_distribution": [
            {"cluster": int(index), "count": int(value)}
            for index, value in distribution.items()
        ]
    }


# ---------------------------------------------------------
# Monitoring (simulated / demonstration only)
# ---------------------------------------------------------

def get_monitoring_results(df):
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint]
    current = df.iloc[midpoint:]

    def psi(expected, actual, bins=10):
        edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)

        if len(edges) < 3:
            return 0.0

        expected_counts = np.histogram(expected, bins=edges)[0]
        actual_counts = np.histogram(actual, bins=edges)[0]

        expected_ratio = np.clip(expected_counts / len(expected), 0.000001, None)
        actual_ratio = np.clip(actual_counts / len(actual), 0.000001, None)

        return float(np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio)))

    drift = []
    for feature in FEATURES:
        value = psi(reference[feature].values, current[feature].values)

        if value < 0.10:
            status = "small change"
        elif value <= 0.25:
            status = "moderate change"
        else:
            status = "significant change"

        drift.append({"feature": feature, "psi": round(value, 4), "status": status})

    return {
        "type": "Demonstration / simulated monitoring",
        "description": "PSI is calculated by comparing the first and second halves of the available dataset. This is not production-time monitoring.",
        "features": drift
    }


# ---------------------------------------------------------
# Master endpoint
# ---------------------------------------------------------

@router.get("/all-results")
def get_all_results():
    df = load_dataset()

    return {
        "status": "success",
        "project": {
            "name": "PlaylistPulse",
            "description": "Spotify Track Intelligence System",
            "dataset": "Spotify Tracks Dataset"
        },
        "dataset": get_dataset_results(df),
        "eda": get_eda_results(df),
        "regression": get_regression_results(df),
        "classification": get_classification_results(df),
        "trees": get_tree_results(df),
        "boosting": get_boosting_results(df),
        "clustering": get_clustering_results(),
        "evaluation": {
            "status": "Evaluation summary file not generated yet",
            "note": "Existing evaluation.py can generate cross-validation and grid-search results."
        },
        "monitoring": get_monitoring_results(df)
    }