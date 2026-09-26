from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILE_2018 = (
    DATA_DIR
    / "2018_param"
    / "HV40_2018_object_points.txt"
)

FILE_2024 = (
    DATA_DIR
    / "2024_param"
    / "HV40_2024_object_points.txt"
)


points_2018 = pd.read_csv(FILE_2018)
points_2024 = pd.read_csv(FILE_2024)


print("2018:")
print(points_2018.head())

print("\n2024:")
print(points_2024.head())


common_points = points_2018.merge(
    points_2024,
    on="GCP_label",
    how="inner",
    suffixes=("_2018", "_2024"),
)


print("\nCommon points:")
print(common_points.to_string(index=False))


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "common_points_2018_2024.csv"

common_points.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved processed data to: {OUTPUT_FILE}")


common_points["delta_X_m"] = (
    common_points["GCP_measured_X_2024"]
    - common_points["GCP_measured_X_2018"]
)

common_points["delta_Y_m"] = (
    common_points["GCP_measured_Y_2024"]
    - common_points["GCP_measured_Y_2018"]
)

common_points["delta_Z_m"] = (
    common_points["GCP_measured_Z_2024"]
    - common_points["GCP_measured_Z_2018"]
)


common_points["displacement_3d_m"] = (
    common_points["delta_X_m"] ** 2
    + common_points["delta_Y_m"] ** 2
    + common_points["delta_Z_m"] ** 2
) ** 0.5


common_points["delta_X_mm"] = common_points["delta_X_m"] * 1000
common_points["delta_Y_mm"] = common_points["delta_Y_m"] * 1000
common_points["delta_Z_mm"] = common_points["delta_Z_m"] * 1000

common_points["displacement_3d_mm"] = (
    common_points["displacement_3d_m"] * 1000
)

result_columns = [
    "GCP_label",
    "delta_X_mm",
    "delta_Y_mm",
    "delta_Z_mm",
    "displacement_3d_mm",
]

print("\nCoordinate changes from 2018 to 2024:")
print(
    common_points[result_columns]
    .round(3)
    .to_string(index=False)
)


DEFORMATION_FILE = (
    PROCESSED_DIR
    / "geodetic_displacement_2018_2024.csv"
)

common_points[result_columns].to_csv(
    DEFORMATION_FILE,
    index=False,
)

print(
    f"\nSaved displacement data to: {DEFORMATION_FILE}"
)


coordinate_sigma_mm = 4.0

delta_sigma_mm = np.sqrt(
    coordinate_sigma_mm**2
    + coordinate_sigma_mm**2
)

confidence_level = 0.95
degrees_of_freedom = 3

chi2_threshold = chi2.ppf(
    confidence_level,
    degrees_of_freedom,
)

displacement_threshold_mm = (
    np.sqrt(chi2_threshold)
    * delta_sigma_mm
)

common_points["chi2_statistic"] = (
    (
        common_points["delta_X_mm"]
        / delta_sigma_mm
    ) ** 2
    + (
        common_points["delta_Y_mm"]
        / delta_sigma_mm
    ) ** 2
    + (
        common_points["delta_Z_mm"]
        / delta_sigma_mm
    ) ** 2
)

common_points["above_95pct_threshold"] = (
    common_points["chi2_statistic"]
    > chi2_threshold
)

print(
    f"\nApproximate uncertainty of a coordinate difference: "
    f"{delta_sigma_mm:.2f} mm"
)

significance_columns = [
    "GCP_label",
    "displacement_3d_mm",
    "chi2_statistic",
    "above_95pct_threshold",
]

print("\nProvisional 3D significance screening:")

print(
    common_points[significance_columns]
    .round(3)
    .to_string(index=False)
)

print(
    f"\n95% 3D displacement threshold: "
    f"{displacement_threshold_mm:.2f} mm"
)

mean_delta_x = common_points["delta_X_mm"].mean()
mean_delta_y = common_points["delta_Y_mm"].mean()
mean_delta_z = common_points["delta_Z_mm"].mean()

mean_displacement = common_points["displacement_3d_mm"].mean()

std_delta_x = common_points["delta_X_mm"].std()
std_delta_y = common_points["delta_Y_mm"].std()
std_delta_z = common_points["delta_Z_mm"].std()

std_displacement = common_points["displacement_3d_mm"].std()

print("\nMean geodetic movement from 2018 to 2024:")

print(f"Mean ΔX: {mean_delta_x:.3f} mm")
print(f"Mean ΔY: {mean_delta_y:.3f} mm")
print(f"Mean ΔZ: {mean_delta_z:.3f} mm")
print(f"Mean 3D displacement: {mean_displacement:.3f} mm")

print("\nPoint-to-point variability:")

print(f"Std ΔX: {std_delta_x:.3f} mm")
print(f"Std ΔY: {std_delta_y:.3f} mm")
print(f"Std ΔZ: {std_delta_z:.3f} mm")
print(f"Std 3D displacement: {std_displacement:.3f} mm")


mean_vector_magnitude = np.sqrt(
    mean_delta_x**2
    + mean_delta_y**2
    + mean_delta_z**2
)

print(
    f"Mean movement vector magnitude: "
    f"{mean_vector_magnitude:.3f} mm"
)

common_points["residual_X_mm"] = (
    common_points["delta_X_mm"] - mean_delta_x
)

common_points["residual_Y_mm"] = (
    common_points["delta_Y_mm"] - mean_delta_y
)

common_points["residual_Z_mm"] = (
    common_points["delta_Z_mm"] - mean_delta_z
)

common_points["residual_3d_mm"] = np.sqrt(
    common_points["residual_X_mm"] ** 2
    + common_points["residual_Y_mm"] ** 2
    + common_points["residual_Z_mm"] ** 2
)

residual_columns = [
    "GCP_label",
    "residual_X_mm",
    "residual_Y_mm",
    "residual_Z_mm",
    "residual_3d_mm",
]

print("\nResidual movement relative to group mean:")

print(
    common_points[residual_columns]
    .round(3)
    .to_string(index=False)
)


number_of_points = len(common_points)

residual_sigma_mm = (
    delta_sigma_mm
    * np.sqrt(1 - 1 / number_of_points)
)

print(
    f"\nApproximate residual component uncertainty: "
    f"{residual_sigma_mm:.2f} mm"
)

residual_threshold_mm = (
    np.sqrt(chi2_threshold)
    * residual_sigma_mm
)

print(
    f"Approximate 95% residual 3D threshold: "
    f"{residual_threshold_mm:.2f} mm"
)

common_points["residual_chi2"] = (
    (
        common_points["residual_X_mm"]
        / residual_sigma_mm
    ) ** 2
    + (
        common_points["residual_Y_mm"]
        / residual_sigma_mm
    ) ** 2
    + (
        common_points["residual_Z_mm"]
        / residual_sigma_mm
    ) ** 2
)

common_points["differential_movement_flag"] = (
    common_points["residual_chi2"]
    > chi2_threshold
)

differential_columns = [
    "GCP_label",
    "residual_3d_mm",
    "residual_chi2",
    "differential_movement_flag",
]

print("\nProvisional differential-movement screening:")

print(
    common_points[differential_columns]
    .round(3)
    .to_string(index=False)
)




























