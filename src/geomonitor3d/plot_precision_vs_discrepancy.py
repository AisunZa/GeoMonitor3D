from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_deformation_analysis_summary.csv"
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


fig, ax = plt.subplots(
    figsize=(7, 6)
)


ax.scatter(
    data[
        "jackknife_displacement_3d_scale_mm"
    ],
    data[
        "robust_vector_difference_3d_mm"
    ],
    s=80,
)


for _, row in data.iterrows():

    ax.annotate(
        row["point_label"],
        (
            row[
                "jackknife_displacement_3d_scale_mm"
            ],
            row[
                "robust_vector_difference_3d_mm"
            ],
        ),
        xytext=(6, 5),
        textcoords="offset points",
    )


maximum_value = max(
    data[
        "jackknife_displacement_3d_scale_mm"
    ].max(),
    data[
        "robust_vector_difference_3d_mm"
    ].max(),
)


ax.plot(
    [0, maximum_value],
    [0, maximum_value],
    linestyle="--",
    linewidth=1.2,
    color="0.4",
    label="External discrepancy = internal scale",
)


ax.set_xlim(
    0,
    maximum_value * 1.08,
)

ax.set_ylim(
    0,
    maximum_value * 1.08,
)


ax.set_xlabel(
    "Internal jackknife displacement "
    "precision scale (mm)"
)

ax.set_ylabel(
    "Robust photogrammetric–geodetic "
    "3D discrepancy (mm)"
)


ax.set_title(
    "Internal Photogrammetric Precision "
    "vs External Displacement Discrepancy"
)


ax.grid(
    True,
    alpha=0.3,
)

ax.legend()


fig.tight_layout()


OUTPUT_FILE = (
    RESULTS_DIR
    / "internal_precision_vs_external_discrepancy.png"
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