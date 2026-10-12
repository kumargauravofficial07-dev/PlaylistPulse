import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "spotify_processed.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "pca_tracks.csv"
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
    print("=== PlaylistPulse PCA ===")

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)

    reduced = pca.fit_transform(X_scaled)

    result = pd.DataFrame(
        reduced,
        columns=["PC1", "PC2"]
    )

    result["popularity"] = df["popularity"].values
    result["popular"] = df["popular"].values

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_FILE, index=False)

    joblib.dump(scaler, MODEL_DIR / "pca_scaler.pkl")
    joblib.dump(pca, MODEL_DIR / "pca_model.pkl")

    print("Original features:", len(FEATURES))
    print("Reduced features: 2")
    print("Explained variance ratio:",
          pca.explained_variance_ratio_)
    print(
        "Total explained variance:",
        round(pca.explained_variance_ratio_.sum(), 4)
    )

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()