import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "raw" / "spotify_tracks.csv"

df = pd.read_csv(DATA_FILE)

print("=== PlaylistPulse EDA ===")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nFirst 5 rows:")
print(df.head())
