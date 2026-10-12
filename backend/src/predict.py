from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

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


def load_model(filename):

    path = MODEL_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return joblib.load(path)


def predict_track(values):

    X = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    result = {}

    regression_path = (
        MODEL_DIR /
        "regression_model.pkl"
    )

    if regression_path.exists():

        regression_model = joblib.load(
            regression_path
        )

        prediction = regression_model.predict(
            X
        )[0]

        result[
            "predicted_popularity"
        ] = round(
            float(prediction),
            2
        )

    classification_path = (
        MODEL_DIR /
        "classification_model.pkl"
    )

    if classification_path.exists():

        classification_model = joblib.load(
            classification_path
        )

        result["popular"] = int(
            classification_model.predict(X)[0]
        )

        result[
            "popular_probability"
        ] = round(
            float(
                classification_model
                .predict_proba(X)[0][1]
            ),
            4
        )

    cluster_model_path = (
        MODEL_DIR /
        "clustering_model.pkl"
    )

    cluster_scaler_path = (
        MODEL_DIR /
        "cluster_scaler.pkl"
    )

    if (
        cluster_model_path.exists()
        and cluster_scaler_path.exists()
    ):

        cluster_model = joblib.load(
            cluster_model_path
        )

        cluster_scaler = joblib.load(
            cluster_scaler_path
        )

        X_scaled = cluster_scaler.transform(X)

        result["cluster"] = int(
            cluster_model.predict(
                X_scaled
            )[0]
        )

    return result