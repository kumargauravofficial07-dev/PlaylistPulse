import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "spotify_tracks.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "spotify_processed.csv"


def load_data():
    return pd.read_csv(RAW_FILE)


def clean_data(df):
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    text_columns = ["artists", "album_name", "track_name"]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")

    return df


def create_targets(df):
    df = df.copy()

    df["popular"] = (df["popularity"] >= 70).astype(int)

    return df


def save_data(df):
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)


def main():
    print("=== PlaylistPulse Preprocessing ===")

    df = load_data()

    print("Original shape:", df.shape)

    df = clean_data(df)

    print("After cleaning:", df.shape)

    df = create_targets(df)

    print("Final shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nPopular distribution:")
    print(df["popular"].value_counts())

    save_data(df)

    print("\nProcessed dataset saved to:")
    print(PROCESSED_FILE)


if __name__ == "__main__":
    main()