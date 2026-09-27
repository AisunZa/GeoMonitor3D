from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_deformation_analysis_summary.csv"
)

POINTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "common_points_2018_2024.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


summary = pd.read_csv(
    SUMMARY_FILE
)

points = pd.read_csv(
    POINTS_FILE
)


data = points.merge(
    summary,
    left_on="GCP_label",
    right_on="point_label",
)


vector_scale_factor = 20


geodetic_dx_m = (
    data["geodetic_delta_X_mm"]
    / 1000
    * vector_scale_factor
)

geodetic_dy_m = (
    data["geodetic_delta_Y_mm"]
    / 1000
    * vector_scale_factor
)


robust_dx_m = (
    data["robust_delta_X_mm"]
    / 1000
    * vector_scale_factor
)

robust_dy_m = (
    data["robust_delta_Y_mm"]
    / 1000
    * vector_scale_factor
)


x_start = data[
    "GCP_measured_X_2018"
]

y_start = data[
    "GCP_measured_Y_2018"
]


fig, ax = plt.subplots(
    figsize=(9, 7)
)


ax.scatter(
    x_start,
    y_start,
    s=45,
)


ax.quiver(
    x_start,
    y_start,
    geodetic_dx_m,
    geodetic_dy_m,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.0045,
    color="black",
    alpha=0.8,
)


ax.quiver(
    x_start,
    y_start,
    robust_dx_m,
    robust_dy_m,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.0025,
    color="red",
    alpha=0.9,
)


for _, row in data.iterrows():

    ax.annotate(
        row["point_label"],
        (
            row["GCP_measured_X_2018"],
            row["GCP_measured_Y_2018"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
    )


all_x = pd.concat(
    [
        x_start,
        x_start + geodetic_dx_m,
        x_start + robust_dx_m,
    ]
)

all_y = pd.concat(
    [
        y_start,
        y_start + geodetic_dy_m,
        y_start + robust_dy_m,
    ]
)


margin = 0.5


ax.set_xlim(
    all_x.min() - margin,
    all_x.max() + margin,
)

ax.set_ylim(
    all_y.min() - margin,
    all_y.max() + margin,
)


from matplotlib.lines import Line2D


legend_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        linewidth=3,
        label="Geodetic",
    ),
    Line2D(
        [0],
        [0],
        color="red",
        linewidth=2,
        label="Robust photogrammetric",
    ),
]


ax.legend(
    handles=legend_handles
)


ax.set_title(
    "Horizontal Displacement: "
    "Robust Photogrammetric vs Geodetic\n"
    "Vectors exaggerated ×20"
)

ax.set_xlabel(
    "X coordinate (m)"
)

ax.set_ylabel(
    "Y coordinate (m)"
)


ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.grid(
    True,
    alpha=0.3,
)


fig.tight_layout()


OUTPUT_FILE = (
    RESULTS_DIR
    / "robust_photogrammetric_vs_geodetic_vectors.png"
)


fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)


print(
    f"Saved figure to:\n"
    f"{OUTPUT_FILE}"
)


plt.show()