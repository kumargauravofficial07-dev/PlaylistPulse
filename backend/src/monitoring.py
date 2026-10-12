import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "spotify_processed.csv"
)


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


def calculate_psi(expected, actual, bins=10):

    edges = np.quantile(
        expected,
        np.linspace(0, 1, bins + 1)
    )

    edges = np.unique(edges)

    if len(edges) < 3:
        return 0.0

    expected_counts = np.histogram(
        expected,
        bins=edges
    )[0]

    actual_counts = np.histogram(
        actual,
        bins=edges
    )[0]

    expected_ratio = (
        expected_counts /
        len(expected)
    )

    actual_ratio = (
        actual_counts /
        len(actual)
    )

    expected_ratio = np.clip(
        expected_ratio,
        0.000001,
        None
    )

    actual_ratio = np.clip(
        actual_ratio,
        0.000001,
        None
    )

    psi = np.sum(
        (
            actual_ratio -
            expected_ratio
        )
        *
        np.log(
            actual_ratio /
            expected_ratio
        )
    )

    return float(psi)


def main():

    print(
        "=== PlaylistPulse Monitoring ==="
    )

    df = pd.read_csv(DATA_FILE)

    midpoint = len(df) // 2

    reference = df.iloc[:midpoint]
    current = df.iloc[midpoint:]

    print("\nFeature Drift:")

    for feature in FEATURES:

        value = calculate_psi(
            reference[feature].values,
            current[feature].values
        )

        print(
            f"{feature:20s}: "
            f"{value:.4f}"
        )

    print(
        "\nInterpretation:"
    )

    print(
        "PSI < 0.10  -> small change"
    )

    print(
        "0.10-0.25   -> moderate change"
    )

    print(
        "PSI > 0.25  -> significant change"
    )


if __name__ == "__main__":
    main()