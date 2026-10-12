import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)


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


def main():

    print("=== PlaylistPulse Evaluation ===")

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df["popular"]

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ]
    )

    scoring = {
        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0
        ),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    summary = {}

    print("\n=== 5-Fold Stratified CV ===")

    for metric in scoring:

        values = results[
            "test_" + metric
        ]

        mean = float(
            np.mean(values)
        )

        std = float(
            np.std(values)
        )

        summary[metric] = {
            "mean": mean,
            "std": std
        }

        print(
            f"{metric}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    print("\n=== Grid Search ===")

    parameter_grid = {
        "model__C": [
            0.01,
            0.1,
            1,
            10
        ]
    }

    grid = GridSearchCV(
        pipeline,
        parameter_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X, y)

    print(
        "Best parameters:",
        grid.best_params_
    )

    print(
        "Best CV F1:",
        round(grid.best_score_, 4)
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "cross_validation": summary,
        "best_parameters": grid.best_params_,
        "best_cv_f1": float(
            grid.best_score_
        )
    }

    with open(
        MODEL_DIR / "evaluation_summary.json",
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        "\nEvaluation report saved."
    )


if __name__ == "__main__":
    main()