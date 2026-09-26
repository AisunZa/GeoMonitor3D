from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILE_2018 = (
    DATA_DIR
    / "2018_param"
    / "HV40_2018_image_points.txt"
)

FILE_2024 = (
    DATA_DIR
    / "2024_param"
    / "HV40_2024_image_points.txt"
)

column_names = [
    "image_name",
    "point_label",
    "image_x",
    "image_y",
]

points_2018 = pd.read_csv(
    FILE_2018,
    header=None,
    names=column_names,
)

points_2024 = pd.read_csv(
    FILE_2024,
    header=None,
    names=column_names,
)

reference_points = [
    "R1",
    "R2",
    "R3",
    "R5",
    "R6",
    "R7",
]

reference_2018 = points_2018[
    points_2018["point_label"].isin(reference_points)
]

reference_2024 = points_2024[
    points_2024["point_label"].isin(reference_points)
]

counts_2018 = (
    reference_2018["point_label"]
    .value_counts()
    .sort_index()
)

counts_2024 = (
    reference_2024["point_label"]
    .value_counts()
    .sort_index()
)

observation_counts = pd.DataFrame(
    {
        "observations_2018": counts_2018,
        "observations_2024": counts_2024,
    }
).fillna(0)

observation_counts = observation_counts.astype(int)

print("\nImage observation counts:")

print(
    observation_counts.to_string()
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "image_observation_counts_2018_2024.csv"
)

observation_counts.to_csv(OUTPUT_FILE)

print(f"\nSaved observation counts to: {OUTPUT_FILE}")

