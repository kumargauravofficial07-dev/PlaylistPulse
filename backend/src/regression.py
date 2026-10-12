import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

TARGET = "popularity"


def load_dataset():
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return model, r2


def main():
    print("=== PlaylistPulse Regression ===")

    X, y = load_dataset()

    print("Dataset shape:", X.shape)
    print("Target:", TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01))
        ]),

        "Elastic Net": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5))
        ])
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        trained_model, r2 = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results[name] = r2
        trained_models[name] = trained_model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save ALL four models separately so the dashboard can show
    # a true comparison across all of them.
    filename_map = {
        "Linear Regression": "regression_linear.pkl",
        "Ridge Regression": "regression_ridge.pkl",
        "Lasso Regression": "regression_lasso.pkl",
        "Elastic Net": "regression_elasticnet.pkl"
    }

    for name, filename in filename_map.items():
        joblib.dump(trained_models[name], MODEL_DIR / filename)
        print(f"Saved {name} -> {filename}")

    # Linear Regression is explicitly kept as the primary/reported
    # model, regardless of which model has the highest R².
    # This is a deliberate project choice (not a bug): Linear
    # Regression is used as the interpretable baseline model.
    joblib.dump(
        trained_models["Linear Regression"],
        MODEL_DIR / "regression_model.pkl"
    )

    print("\n=== Primary Regression Model ===")
    print("Model: Linear Regression (fixed choice)")
    print("R2:", round(results["Linear Regression"], 4))

    print("\n=== All Regression R² Scores ===")
    for name, r2 in results.items():
        print(f"{name}: {round(r2, 4)}")

    print("\nAll regression models saved successfully.")


if __name__ == "__main__":
    main()