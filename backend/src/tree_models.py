import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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


def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )
    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print("\n" + name)
    print("-" * 40)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    return model


def main():

    print("=== PlaylistPulse Tree Models ===")

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

    decision_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    random_forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    tree_model = evaluate_model(
        "Decision Tree",
        decision_tree,
        X_train,
        X_test,
        y_train,
        y_test
    )

    forest_model = evaluate_model(
        "Random Forest",
        random_forest,
        X_train,
        X_test,
        y_train,
        y_test
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        tree_model,
        MODEL_DIR / "decision_tree.pkl"
    )

    joblib.dump(
        forest_model,
        MODEL_DIR / "random_forest.pkl"
    )

    joblib.dump(
        FEATURES,
        MODEL_DIR / "tree_features.pkl"
    )

    print("\n=== Random Forest OOB ===")
    print(round(random_forest.oob_score_, 4))

    print("\nFeature Importance:")

    importance = pd.Series(
        random_forest.feature_importances_,
        index=FEATURES
    ).sort_values(
        ascending=False
    )

    print(importance)

    print("\nModels saved successfully.")


if __name__ == "__main__":
    main()