from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


standard = data[
    "standard_vector_difference_3d_mm"
].to_numpy()

robust = data[
    "robust_vector_difference_3d_mm"
].to_numpy()


standard_rmse = np.sqrt(
    np.mean(
        standard**2
    )
)

robust_rmse = np.sqrt(
    np.mean(
        robust**2
    )
)


x = np.arange(
    len(data)
)

bar_width = 0.36


fig, ax = plt.subplots(
    figsize=(9, 5.5)
)


standard_bars = ax.bar(
    x - bar_width / 2,
    standard,
    width=bar_width,
    label="Standard triangulation",
    color="0.65",
)


robust_bars = ax.bar(
    x + bar_width / 2,
    robust,
    width=bar_width,
    label="Robust triangulation",
    color="firebrick",
)


ax.axhline(
    standard_rmse,
    color="0.35",
    linestyle="--",
    linewidth=1.3,
    label=(
        f"Standard RMSE "
        f"({standard_rmse:.2f} mm)"
    ),
)


ax.axhline(
    robust_rmse,
    color="firebrick",
    linestyle=":",
    linewidth=1.8,
    label=(
        f"Robust RMSE "
        f"({robust_rmse:.2f} mm)"
    ),
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    data["point_label"]
)


ax.set_xlabel(
    "Monitoring point"
)

ax.set_ylabel(
    "3D displacement-vector discrepancy (mm)"
)


ax.set_title(
    "Standard vs Robust Photogrammetric "
    "Displacement Agreement"
)


ax.grid(
    axis="y",
    alpha=0.25,
)


ax.legend()


ax.bar_label(
    standard_bars,
    fmt="%.2f",
    padding=3,
    fontsize=8,
)

ax.bar_label(
    robust_bars,
    fmt="%.2f",
    padding=3,
    fontsize=8,
)


fig.tight_layout()


OUTPUT_FILE = (
    RESULTS_DIR
    / "standard_vs_robust_displacement_discrepancy.png"
)


fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)


print(
    f"Standard RMSE: "
    f"{standard_rmse:.3f} mm"
)

print(
    f"Robust RMSE: "
    f"{robust_rmse:.3f} mm"
)

print(
    f"Saved figure to:\n"
    f"{OUTPUT_FILE}"
)


plt.show()