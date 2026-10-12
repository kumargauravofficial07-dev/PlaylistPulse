from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.predict import FEATURES, predict_track


router = APIRouter(prefix="/api", tags=["Track Explorer"])
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "spotify_processed.csv"


class TrackPredictionInput(BaseModel):
    track_name: str = Field(default="", max_length=200)
    artists: str = Field(default="", max_length=300)
    duration_ms: float = Field(gt=0, le=3_600_000)
    danceability: float = Field(ge=0, le=1)
    energy: float = Field(ge=0, le=1)
    loudness: float = Field(ge=-100, le=20)
    speechiness: float = Field(ge=0, le=1)
    acousticness: float = Field(ge=0, le=1)
    instrumentalness: float = Field(ge=0, le=1)
    liveness: float = Field(ge=0, le=1)
    valence: float = Field(ge=0, le=1)
    tempo: float = Field(gt=0, le=400)


@lru_cache(maxsize=1)
def load_dataset():
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="The processed Spotify dataset was not found.")
    return pd.read_csv(DATA_FILE)


def track_result(row):
    return {
        "track_id": str(row.get("track_id", "")),
        "track_name": str(row.get("track_name", "")),
        "artists": str(row.get("artists", "")),
        "track_genre": str(row.get("track_genre", "")),
        "popularity": float(row.get("popularity", 0)),
        "features": {feature: float(row[feature]) for feature in FEATURES},
    }


@router.get("/genres")
def get_genres():
    df = load_dataset()
    return sorted(df["track_genre"].dropna().astype(str).unique().tolist())


@router.get("/tracks")
def get_tracks(
    query: str = Query(default="", max_length=100),
    genre: str = Query(default="", max_length=100),
    limit: int = Query(default=12, ge=1, le=50),
):
    df = load_dataset()
    if genre:
        df = df[df["track_genre"].astype(str) == genre]
    if query.strip():
        text = query.strip()
        matches_name = df["track_name"].astype(str).str.contains(text, case=False, regex=False, na=False)
        matches_artist = df["artists"].astype(str).str.contains(text, case=False, regex=False, na=False)
        df = df[matches_name | matches_artist]

    df = df.sort_values("popularity", ascending=False).head(limit)
    return {"tracks": [track_result(row) for _, row in df.iterrows()]}


@router.post("/predict")
def predict_from_track(payload: TrackPredictionInput):
    values = [getattr(payload, feature) for feature in FEATURES]
    try:
        result = predict_track(values)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The prediction models could not process these values. Check the backend terminal for details.",
        ) from exc

    if not result:
        raise HTTPException(status_code=503, detail="No trained prediction models were found.")
    return result
