from fastapi import APIRouter
import pandas as pd
from pathlib import Path

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "spotify_processed.csv"


@router.get("/analytics")
def get_analytics():
    df = pd.read_csv(DATA_PATH)

    numerical_columns = [
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "popularity"
    ]

    statistics = {}

    for column in numerical_columns:
        statistics[column] = {
            "mean": round(float(df[column].mean()), 2),
            "median": round(float(df[column].median()), 2),
            "min": round(float(df[column].min()), 2),
            "max": round(float(df[column].max()), 2)
        }

    genre_counts = (
        df["track_genre"]
        .value_counts()
        .head(15)
        .to_dict()
    )

    return {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns))
        },
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "statistics": statistics,
        "top_genres": genre_counts
    }