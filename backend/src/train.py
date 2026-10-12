import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "preprocessing.py",
    "regression.py",
    "classification.py",
    "tree_models.py",
    "boosting.py",
    "clustering.py",
    "features.py",
    "evaluation.py",
    "monitoring.py"
]


def main():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PLAYLISTPULSE COMPLETE ML PIPELINE"
    )

    print(
        "=" * 60
    )

    for script in SCRIPTS:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "RUNNING:",
            script
        )

        print(
            "=" * 60
        )

        result = subprocess.run(
            [
                sys.executable,
                str(BASE_DIR / script)
            ]
        )

        if result.returncode != 0:

            print(
                "\nPipeline stopped."
            )

            print(
                "Failed stage:",
                script
            )

            sys.exit(
                result.returncode
            )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PLAYLISTPULSE PIPELINE COMPLETED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()