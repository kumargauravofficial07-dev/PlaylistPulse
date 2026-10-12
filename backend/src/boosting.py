import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "spotify_processed.csv"
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


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + name)
    print("-" * 40)

    print(
        f"Accuracy  : "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Precision : "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall    : "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1 Score  : "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )

    return model


def main():

    print("=== PlaylistPulse Boosting ===")

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df["popular"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    ada = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=42
    )

    gradient = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    ada_model = evaluate_model(
        "AdaBoost",
        ada,
        X_train,
        X_test,
        y_train,
        y_test
    )

    gradient_model = evaluate_model(
        "Gradient Boosting",
        gradient,
        X_train,
        X_test,
        y_train,
        y_test
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        ada_model,
        MODEL_DIR / "adaboost_model.pkl"
    )

    joblib.dump(
        gradient_model,
        MODEL_DIR / "gradient_boosting_model.pkl"
    )

    try:

        from xgboost import XGBClassifier

        positive_weight = (
            (y_train == 0).sum() /
            (y_train == 1).sum()
        )

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=positive_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

        xgb_model = evaluate_model(
            "XGBoost",
            xgb,
            X_train,
            X_test,
            y_train,
            y_test
        )

        joblib.dump(
            xgb_model,
            MODEL_DIR / "xgboost_model.pkl"
        )

    except Exception as error:

        print("\nXGBoost could not be trained.")
        print(error)

    joblib.dump(
        FEATURES,
        MODEL_DIR / "boosting_features.pkl"
    )

    print("\nBoosting models saved.")


if __name__ == "__main__":
    main()