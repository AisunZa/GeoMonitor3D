from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "displacement_comparison_all_points.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


data = pd.read_csv(
    INPUT_FILE
)


data["total_observations"] = (
    data["observations_2018"]
    + data["observations_2024"]
)


correlation, p_value = pearsonr(
    data["total_observations"],
    data["vector_difference_3d_mm"],
)


print(
    "\nObservation-count analysis:"
)

print(
    data[
        [
            "point_label",
            "observations_2018",
            "observations_2024",
            "total_observations",
            "vector_difference_3d_mm",
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


print(
    f"\nPearson correlation: "
    f"{correlation:.3f}"
)

print(
    f"p-value: "
    f"{p_value:.3f}"
)


fig, ax = plt.subplots(
    figsize=(7, 5)
)


ax.scatter(
    data["total_observations"],
    data["vector_difference_3d_mm"],
    s=70,
)


for _, row in data.iterrows():
    ax.annotate(
        row["point_label"],
        (
            row["total_observations"],
            row["vector_difference_3d_mm"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
    )


ax.set_title(
    "Observation Count vs "
    "3D Displacement-Vector Discrepancy"
)

ax.set_xlabel(
    "Total image observations "
    "(2018 + 2024)"
)

ax.set_ylabel(
    "3D vector discrepancy (mm)"
)

ax.grid(
    True,
    alpha=0.3,
)


fig.tight_layout()


OUTPUT_FILE = (
    RESULTS_DIR
    / "observation_count_vs_discrepancy.png"
)

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)


print(
    f"\nSaved figure to: "
    f"{OUTPUT_FILE}"
)

plt.show()