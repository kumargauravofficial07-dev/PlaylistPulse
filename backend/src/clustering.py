from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "spotify_processed.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

    print("=== PlaylistPulse Fast Clustering ===")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(
        scaler,
        MODEL_DIR / "cluster_scaler.pkl"
    )

    print(f"Dataset: {len(df):,} tracks")
    print("Features:", len(FEATURES))

    rng = np.random.RandomState(42)

    sample_size = min(5000, len(df))

    sample_indices = rng.choice(
        len(df),
        size=sample_size,
        replace=False
    )

    X_sample = X_scaled[sample_indices]

    print("\n=== Fast K-Means Search ===")

    best_k = 2
    best_score = -1

    for k in range(2, 7):

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=2048,
            n_init=3,
            max_iter=100
        )

        labels = model.fit_predict(X_sample)

        score = silhouette_score(
            X_sample,
            labels
        )

        print(
            f"k={k}  "
            f"Silhouette={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_k = k

    print(
        f"\nBest k: {best_k}"
    )

    print("\n=== Training Final MiniBatch K-Means ===")

    kmeans = MiniBatchKMeans(
        n_clusters=best_k,
        random_state=42,
        batch_size=4096,
        n_init=5,
        max_iter=150
    )

    full_labels = kmeans.fit_predict(X_scaled)

    df["cluster"] = full_labels

    joblib.dump(
        kmeans,
        MODEL_DIR / "clustering_model.pkl"
    )

    output_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "clustered_tracks.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n=== Cluster Distribution ===")

    print(
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    print("\n=== Hierarchical Clustering Preview ===")

    hierarchical_size = min(
        2000,
        len(X_sample)
    )

    hierarchical_indices = rng.choice(
        len(X_sample),
        size=hierarchical_size,
        replace=False
    )

    X_hierarchical = X_sample[
        hierarchical_indices
    ]

    hierarchical = AgglomerativeClustering(
        n_clusters=best_k,
        linkage="ward"
    )

    hierarchical_labels = hierarchical.fit_predict(
        X_hierarchical
    )

    hierarchical_score = silhouette_score(
        X_hierarchical,
        hierarchical_labels
    )

    print(
        f"Agglomerative silhouette: "
        f"{hierarchical_score:.4f}"
    )

    print("\n=== DBSCAN Preview ===")

    dbscan_size = min(
        5000,
        len(X_sample)
    )

    dbscan_indices = rng.choice(
        len(X_sample),
        size=dbscan_size,
        replace=False
    )

    X_dbscan = X_sample[
        dbscan_indices
    ]

    dbscan = DBSCAN(
        eps=1.5,
        min_samples=10,
        n_jobs=-1
    )

    dbscan_labels = dbscan.fit_predict(
        X_dbscan
    )

    unique_labels = set(dbscan_labels)

    clusters_found = len(
        unique_labels - {-1}
    )

    noise_count = np.sum(
        dbscan_labels == -1
    )

    print(
        f"DBSCAN clusters: {clusters_found}"
    )

    print(
        f"DBSCAN noise points: {noise_count}"
    )

    print("\n=== Clustering Completed ===")

    print(
        f"Saved clustered dataset to:\n"
        f"{output_path}"
    )

    print(
        f"Saved K-Means model to:\n"
        f"{MODEL_DIR / 'clustering_model.pkl'}"
    )


if __name__ == "__main__":
    main()