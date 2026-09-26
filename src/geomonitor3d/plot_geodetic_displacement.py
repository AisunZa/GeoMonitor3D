from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "common_points_2018_2024.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

points = pd.read_csv(INPUT_FILE)

points["delta_X_m"] = (
    points["GCP_measured_X_2024"]
    - points["GCP_measured_X_2018"]
)

points["delta_Y_m"] = (
    points["GCP_measured_Y_2024"]
    - points["GCP_measured_Y_2018"]
)

vector_scale_factor = 20

points["plot_delta_X"] = (
    points["delta_X_m"] * vector_scale_factor
)

points["plot_delta_Y"] = (
    points["delta_Y_m"] * vector_scale_factor
)

points["arrow_end_X"] = (
    points["GCP_measured_X_2018"]
    + points["plot_delta_X"]
)

points["arrow_end_Y"] = (
    points["GCP_measured_Y_2018"]
    + points["plot_delta_Y"]
)

fig, ax = plt.subplots(figsize=(8, 7))

ax.scatter(
    points["GCP_measured_X_2018"],
    points["GCP_measured_Y_2018"],
    s=50,
)

ax.quiver(
    points["GCP_measured_X_2018"],
    points["GCP_measured_Y_2018"],
    points["plot_delta_X"],
    points["plot_delta_Y"],
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.004,
)

all_x = pd.concat(
    [
        points["GCP_measured_X_2018"],
        points["arrow_end_X"],
    ]
)

all_y = pd.concat(
    [
        points["GCP_measured_Y_2018"],
        points["arrow_end_Y"],
    ]
)

x_margin = 0.5
y_margin = 0.5

ax.set_xlim(
    all_x.min() - x_margin,
    all_x.max() + x_margin,
)

ax.set_ylim(
    all_y.min() - y_margin,
    all_y.max() + y_margin,
)


for _, row in points.iterrows():
    ax.annotate(
        row["GCP_label"],
        (
            row["GCP_measured_X_2018"],
            row["GCP_measured_Y_2018"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
    )
    
ax.set_title(
    "Horizontal Geodetic Displacement, 2018–2024\n"
    "Displacement arrows exaggerated ×20"
)

ax.set_xlabel("X coordinate (m)")
ax.set_ylabel("Y coordinate (m)")

ax.grid(True, alpha=0.3)
ax.set_aspect("equal", adjustable="box")

OUTPUT_FILE = (
    RESULTS_DIR
    / "horizontal_geodetic_displacement_2018_2024.png"
)

fig.tight_layout()
fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved figure to: {OUTPUT_FILE}")

plt.show()