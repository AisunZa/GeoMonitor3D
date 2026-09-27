from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares, root
from scipy.stats import chi2


# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BASE_EPOCH = "2018"
FOLLOWUP_EPOCH = "2024"
EPOCHS = (BASE_EPOCH, FOLLOWUP_EPOCH)

TARGET_POINTS = ("R1", "R2", "R3", "R5", "R6", "R7")

# Robust-estimation settings.
HUBER_F_SCALE_PX = 1.0
HUBER_SENSITIVITY_SCALES_PX = (0.5, 1.0, 2.0)

# Leave-one-out screening.
LOO_MODIFIED_Z_THRESHOLD = 3.5

# Geodetic uncertainty sensitivity.
# The dataset documentation reports approximately 4 mm coordinate accuracy.
# Treating that value as 1-sigma is a provisional modeling assumption.
GEODETIC_SIGMA_PER_EPOCH_MM = 4.0
GEODETIC_SIGMA_SCENARIOS_MM = (1.0, 2.0, 3.0, 4.0)

CHI2_CONFIDENCE = 0.95
CHI2_DF = 3
CHI2_THRESHOLD_95 = chi2.ppf(CHI2_CONFIDENCE, df=CHI2_DF)


# -----------------------------------------------------------------------------
# File/path helpers
# -----------------------------------------------------------------------------

def get_epoch_paths(epoch: str) -> dict[str, Path]:
    """Return all input paths for one epoch."""

    param_dir = DATA_DIR / f"{epoch}_param"

    return {
        "param_dir": param_dir,
        "camera": param_dir / f"HV40_{epoch}_calibrated_camera_parameters.txt",
        "image_points": param_dir / f"HV40_{epoch}_image_points.txt",
        "object_points": param_dir / f"HV40_{epoch}_object_points.txt",
        "offset": param_dir / f"HV40_{epoch}_offset.xyz",
    }


def validate_epoch_paths(paths: dict[str, Path], epoch: str) -> None:
    """Fail early if any required epoch input is missing."""

    required = ("camera", "image_points", "object_points", "offset")
    missing = [str(paths[name]) for name in required if not paths[name].exists()]

    if missing:
        raise FileNotFoundError(
            f"Missing required input file(s) for epoch {epoch}:\n"
            + "\n".join(missing)
        )


# -----------------------------------------------------------------------------
# Input readers
# -----------------------------------------------------------------------------

def load_camera_parameters(file_path: Path) -> dict[str, dict]:
    """
    Read a Pix4D calibrated-camera-parameter file.

    Returns
    -------
    dict
        Dictionary keyed by image stem, e.g. ``DSC02929``.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    cameras: dict[str, dict] = {}
    i = 0

    while i < len(lines):
        parts = lines[i].split()

        is_camera_line = (
            len(parts) == 3
            and parts[0].lower().endswith((".jpg", ".jpeg"))
        )

        if not is_camera_line:
            i += 1
            continue

        if i + 9 >= len(lines):
            raise ValueError(
                f"Incomplete camera block beginning at line {i + 1} "
                f"in {file_path}."
            )

        image_name = Path(parts[0]).stem
        image_width = int(parts[1])
        image_height = int(parts[2])

        K = np.array(
            [
                [float(value) for value in lines[i + 1].split()],
                [float(value) for value in lines[i + 2].split()],
                [float(value) for value in lines[i + 3].split()],
            ],
            dtype=float,
        )

        radial = np.array(
            [float(value) for value in lines[i + 4].split()],
            dtype=float,
        )

        tangential = np.array(
            [float(value) for value in lines[i + 5].split()],
            dtype=float,
        )

        camera_position = np.array(
            [float(value) for value in lines[i + 6].split()],
            dtype=float,
        )

        rotation = np.array(
            [
                [float(value) for value in lines[i + 7].split()],
                [float(value) for value in lines[i + 8].split()],
                [float(value) for value in lines[i + 9].split()],
            ],
            dtype=float,
        )

        if K.shape != (3, 3):
            raise ValueError(f"Invalid K matrix for {image_name} in {file_path}.")
        if radial.size != 3:
            raise ValueError(
                f"Expected 3 radial coefficients for {image_name}, "
                f"found {radial.size}."
            )
        if tangential.size != 2:
            raise ValueError(
                f"Expected 2 tangential coefficients for {image_name}, "
                f"found {tangential.size}."
            )
        if camera_position.size != 3:
            raise ValueError(
                f"Expected a 3D camera position for {image_name}."
            )
        if rotation.shape != (3, 3):
            raise ValueError(
                f"Expected a 3x3 rotation matrix for {image_name}."
            )

        cameras[image_name] = {
            "width": image_width,
            "height": image_height,
            "K": K,
            "radial": radial,
            "tangential": tangential,
            "camera_position": camera_position,
            "rotation": rotation,
        }

        i += 10

    if not cameras:
        raise ValueError(f"No camera blocks were read from {file_path}.")

    return cameras


def load_image_points(file_path: Path) -> pd.DataFrame:
    """Read the headerless image-point file."""

    image_points = pd.read_csv(
        file_path,
        header=None,
        names=[
            "image_name",
            "point_label",
            "image_x",
            "image_y",
        ],
    )

    image_points["image_name"] = image_points["image_name"].map(
        lambda value: Path(str(value).strip()).stem
    )
    image_points["point_label"] = (
        image_points["point_label"].astype(str).str.strip()
    )
    image_points["image_x"] = pd.to_numeric(
        image_points["image_x"],
        errors="raise",
    )
    image_points["image_y"] = pd.to_numeric(
        image_points["image_y"],
        errors="raise",
    )

    return image_points


def load_object_points(file_path: Path) -> pd.DataFrame:
    """Read surveyed/reference object coordinates."""

    object_points = pd.read_csv(file_path)

    required_columns = {
        "GCP_label",
        "GCP_measured_X",
        "GCP_measured_Y",
        "GCP_measured_Z",
    }

    missing = required_columns.difference(object_points.columns)
    if missing:
        raise ValueError(
            f"Missing required object-point columns in {file_path}: "
            f"{sorted(missing)}"
        )

    object_points["GCP_label"] = (
        object_points["GCP_label"].astype(str).str.strip()
    )

    for column in (
        "GCP_measured_X",
        "GCP_measured_Y",
        "GCP_measured_Z",
    ):
        object_points[column] = pd.to_numeric(
            object_points[column],
            errors="raise",
        )

    return object_points


def load_offset(file_path: Path) -> np.ndarray:
    """Read the 3D epoch offset used by the local camera-coordinate system."""

    offset = np.asarray(np.loadtxt(file_path), dtype=float).reshape(-1)

    if offset.size != 3:
        raise ValueError(
            f"Expected exactly 3 offset values in {file_path}, "
            f"found {offset.size}."
        )

    return offset


def load_epoch_data(epoch: str) -> dict:
    """Load all inputs required for one epoch once."""

    paths = get_epoch_paths(epoch)
    validate_epoch_paths(paths, epoch)

    return {
        "epoch": epoch,
        "paths": paths,
        "cameras": load_camera_parameters(paths["camera"]),
        "image_points": load_image_points(paths["image_points"]),
        "object_points": load_object_points(paths["object_points"]),
        "offset": load_offset(paths["offset"]),
    }


def get_point_observations(
    epoch_data: dict,
    point_label: str,
) -> pd.DataFrame:
    """Return all image observations for one monitoring point."""

    return epoch_data["image_points"][
        epoch_data["image_points"]["point_label"] == point_label
    ].copy()


# -----------------------------------------------------------------------------
# Camera model
# -----------------------------------------------------------------------------

def distort_normalized(
    x: float,
    y: float,
    radial: np.ndarray,
    tangential: np.ndarray,
) -> tuple[float, float]:
    """Apply radial and tangential distortion to normalized coordinates."""

    k1, k2, k3 = radial
    t1, t2 = tangential

    r_squared = x**2 + y**2

    radial_factor = (
        1
        + k1 * r_squared
        + k2 * r_squared**2
        + k3 * r_squared**3
    )

    x_distorted = (
        x * radial_factor
        + 2 * t1 * x * y
        + t2 * (r_squared + 2 * x**2)
    )

    y_distorted = (
        y * radial_factor
        + t1 * (r_squared + 2 * y**2)
        + 2 * t2 * x * y
    )

    return float(x_distorted), float(y_distorted)


def undistort_pixel(
    u: float,
    v: float,
    K: np.ndarray,
    radial: np.ndarray,
    tangential: np.ndarray,
) -> np.ndarray:
    """
    Convert a measured distorted pixel to undistorted normalized coordinates.
    """

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x_distorted = (u - cx) / fx
    y_distorted = (v - cy) / fy

    def equations(unknowns: np.ndarray) -> np.ndarray:
        x, y = unknowns
        predicted_x, predicted_y = distort_normalized(
            x,
            y,
            radial,
            tangential,
        )
        return np.array(
            [
                predicted_x - x_distorted,
                predicted_y - y_distorted,
            ],
            dtype=float,
        )

    solution = root(
        equations,
        [x_distorted, y_distorted],
    )

    if not solution.success or not np.all(np.isfinite(solution.x)):
        raise RuntimeError(
            "Distortion inversion did not converge for "
            f"pixel ({u}, {v})."
        )

    return np.asarray(solution.x, dtype=float)


def project_local_point_to_pixel(
    xyz_local: np.ndarray,
    camera: dict,
) -> np.ndarray:
    """Project a local 3D point into one camera image."""

    xyz_camera = camera["rotation"] @ (
        xyz_local - camera["camera_position"]
    )

    if abs(xyz_camera[2]) < 1e-12:
        raise ZeroDivisionError(
            "Point projects with near-zero camera-space depth."
        )

    x_ideal = xyz_camera[0] / xyz_camera[2]
    y_ideal = xyz_camera[1] / xyz_camera[2]

    x_distorted, y_distorted = distort_normalized(
        x_ideal,
        y_ideal,
        camera["radial"],
        camera["tangential"],
    )

    K = camera["K"]
    u = K[0, 0] * x_distorted + K[0, 2]
    v = K[1, 1] * y_distorted + K[1, 2]

    return np.array([u, v], dtype=float)


# -----------------------------------------------------------------------------
# Triangulation and image-space diagnostics
# -----------------------------------------------------------------------------

def validate_observations(
    observations: pd.DataFrame,
    cameras: dict[str, dict],
    point_label: str | None = None,
    epoch: str | None = None,
) -> None:
    """Validate observation count and camera availability."""

    context = ""
    if point_label is not None or epoch is not None:
        context = f" for point={point_label}, epoch={epoch}"

    if len(observations) < 2:
        raise ValueError(
            f"At least 2 image observations are required{context}; "
            f"found {len(observations)}."
        )

    missing_cameras = sorted(
        set(observations["image_name"]) - set(cameras.keys())
    )

    if missing_cameras:
        raise KeyError(
            f"Missing camera parameters{context}: {missing_cameras}"
        )


def build_object_space_rays(
    observations: pd.DataFrame,
    cameras: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Build camera origins and unit object-space ray directions."""

    validate_observations(observations, cameras)

    ray_origins = []
    ray_directions = []

    for _, observation in observations.iterrows():
        image_name = observation["image_name"]
        camera = cameras[image_name]

        x_undistorted, y_undistorted = undistort_pixel(
            observation["image_x"],
            observation["image_y"],
            camera["K"],
            camera["radial"],
            camera["tangential"],
        )

        direction_camera = np.array(
            [x_undistorted, y_undistorted, 1.0],
            dtype=float,
        )

        direction_object = camera["rotation"].T @ direction_camera
        direction_norm = np.linalg.norm(direction_object)

        if direction_norm <= 1e-15:
            raise ValueError(
                f"Degenerate ray direction for image {image_name}."
            )

        direction_object /= direction_norm

        ray_origins.append(camera["camera_position"])
        ray_directions.append(direction_object)

    return (
        np.asarray(ray_origins, dtype=float),
        np.asarray(ray_directions, dtype=float),
    )


def solve_multi_ray_intersection(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Solve the least-squares closest point to multiple 3D viewing rays.

    Returns the local 3D point plus camera-ray geometry diagnostics.
    """

    identity = np.eye(3)
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)

    for camera_position, direction in zip(
        ray_origins,
        ray_directions,
    ):
        projection = identity - np.outer(direction, direction)
        A += projection
        b += projection @ camera_position

    normalized_A = A / len(ray_origins)
    geometry_eigenvalues = np.linalg.eigvalsh(normalized_A)

    lambda_min = float(geometry_eigenvalues[0])
    lambda_max = float(geometry_eigenvalues[-1])

    if lambda_min <= 1e-12:
        raise np.linalg.LinAlgError(
            "Triangulation geometry is singular or nearly singular."
        )

    xyz_local = np.linalg.solve(A, b)

    geometry = {
        "geometry_lambda_min": float(geometry_eigenvalues[0]),
        "geometry_lambda_mid": float(geometry_eigenvalues[1]),
        "geometry_lambda_max": float(geometry_eigenvalues[2]),
        "geometry_condition_number": lambda_max / lambda_min,
    }

    return xyz_local, geometry


def triangulate_from_observations(
    observations: pd.DataFrame,
    cameras: dict[str, dict],
    return_geometry: bool = False,
):
    """Triangulate a local 3D point from image observations."""

    ray_origins, ray_directions = build_object_space_rays(
        observations,
        cameras,
    )

    xyz_local, geometry = solve_multi_ray_intersection(
        ray_origins,
        ray_directions,
    )

    if return_geometry:
        return xyz_local, geometry

    return xyz_local


def compute_reprojection_diagnostics(
    xyz_local: np.ndarray,
    observations: pd.DataFrame,
    cameras: dict[str, dict],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute measured-minus-predicted reprojection residuals."""

    records = []

    for _, observation in observations.iterrows():
        image_name = observation["image_name"]

        predicted_pixel = project_local_point_to_pixel(
            xyz_local,
            cameras[image_name],
        )

        measured_pixel = np.array(
            [observation["image_x"], observation["image_y"]],
            dtype=float,
        )

        residual = measured_pixel - predicted_pixel
        error_px = float(np.linalg.norm(residual))

        records.append(
            {
                "image_name": image_name,
                "residual_u_px": float(residual[0]),
                "residual_v_px": float(residual[1]),
                "error_px": error_px,
            }
        )

    residual_df = pd.DataFrame(records)
    errors = residual_df["error_px"].to_numpy(dtype=float)

    summary = {
        "reprojection_mean_px": float(np.mean(errors)),
        "reprojection_rmse_px": float(np.sqrt(np.mean(errors**2))),
        "reprojection_max_px": float(np.max(errors)),
    }

    return residual_df, summary


def robust_triangulate_from_observations(
    observations: pd.DataFrame,
    cameras: dict[str, dict],
    initial_xyz: np.ndarray,
    f_scale: float = HUBER_F_SCALE_PX,
) -> np.ndarray:
    """
    Refine a local 3D point by robustly minimizing pixel residual components.

    The surveyed/reference coordinate is not used in this estimation.
    """

    validate_observations(observations, cameras)

    if f_scale <= 0:
        raise ValueError("Huber f_scale must be positive.")

    def residual_function(xyz_local: np.ndarray) -> np.ndarray:
        residuals = []

        for _, observation in observations.iterrows():
            image_name = observation["image_name"]

            predicted_pixel = project_local_point_to_pixel(
                xyz_local,
                cameras[image_name],
            )

            measured_pixel = np.array(
                [observation["image_x"], observation["image_y"]],
                dtype=float,
            )

            residual = measured_pixel - predicted_pixel
            residuals.extend(residual)

        return np.asarray(residuals, dtype=float)

    result = least_squares(
        residual_function,
        np.asarray(initial_xyz, dtype=float),
        loss="huber",
        f_scale=f_scale,
    )

    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            "Robust triangulation failed to converge: "
            f"{result.message}"
        )

    return np.asarray(result.x, dtype=float)


# -----------------------------------------------------------------------------
# Baseline triangulation and displacement comparison
# -----------------------------------------------------------------------------

def get_reference_coordinate(
    object_points: pd.DataFrame,
    target_point: str,
    epoch: str,
) -> np.ndarray:
    """Return the unique surveyed/reference XYZ coordinate for a point."""

    reference = object_points[
        object_points["GCP_label"] == target_point
    ]

    if len(reference) != 1:
        raise ValueError(
            f"Expected exactly one surveyed/reference coordinate for "
            f"{target_point} in {epoch}; found {len(reference)}."
        )

    return np.array(
        [
            reference.iloc[0]["GCP_measured_X"],
            reference.iloc[0]["GCP_measured_Y"],
            reference.iloc[0]["GCP_measured_Z"],
        ],
        dtype=float,
    )


def triangulate_epoch(
    epoch: str,
    target_point: str,
    epoch_data: dict,
) -> tuple[dict, pd.DataFrame]:
    """
    Triangulate one target point for one epoch and compare it to the
    surveyed/reference coordinate.
    """

    cameras = epoch_data["cameras"]
    offset = epoch_data["offset"]

    observations = get_point_observations(
        epoch_data,
        target_point,
    )

    validate_observations(
        observations,
        cameras,
        point_label=target_point,
        epoch=epoch,
    )

    xyz_local_photo, geometry = triangulate_from_observations(
        observations,
        cameras,
        return_geometry=True,
    )

    xyz_absolute_photo = xyz_local_photo + offset

    xyz_reference = get_reference_coordinate(
        epoch_data["object_points"],
        target_point,
        epoch,
    )

    coordinate_difference_mm = (
        xyz_absolute_photo - xyz_reference
    ) * 1000.0

    residual_df, reprojection_summary = compute_reprojection_diagnostics(
        xyz_local_photo,
        observations,
        cameras,
    )

    residual_df.insert(0, "epoch", epoch)
    residual_df.insert(0, "point_label", target_point)

    result = {
        "point_label": target_point,
        "epoch": epoch,
        "observations": len(observations),
        "photo_X_local_m": xyz_local_photo[0],
        "photo_Y_local_m": xyz_local_photo[1],
        "photo_Z_local_m": xyz_local_photo[2],
        "photo_X_m": xyz_absolute_photo[0],
        "photo_Y_m": xyz_absolute_photo[1],
        "photo_Z_m": xyz_absolute_photo[2],
        "reference_X_m": xyz_reference[0],
        "reference_Y_m": xyz_reference[1],
        "reference_Z_m": xyz_reference[2],
        "photo_minus_ref_X_mm": coordinate_difference_mm[0],
        "photo_minus_ref_Y_mm": coordinate_difference_mm[1],
        "photo_minus_ref_Z_mm": coordinate_difference_mm[2],
        "photo_minus_ref_3d_mm": np.linalg.norm(
            coordinate_difference_mm
        ),
        **geometry,
        **reprojection_summary,
    }

    return result, residual_df


def run_baseline_triangulation(
    epoch_data_by_epoch: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run standard triangulation for every requested point and epoch."""

    results = []
    residual_tables = []

    for target_point in TARGET_POINTS:
        for epoch in EPOCHS:
            result, residual_df = triangulate_epoch(
                epoch,
                target_point,
                epoch_data_by_epoch[epoch],
            )
            results.append(result)
            residual_tables.append(residual_df)

    return (
        pd.DataFrame(results),
        pd.concat(residual_tables, ignore_index=True),
    )


def xyz_from_row(
    row: pd.Series,
    prefix: str,
    suffix: str = "_m",
) -> np.ndarray:
    """Build XYZ vector from columns such as photo_X_m/photo_Y_m/photo_Z_m."""

    return np.array(
        [
            row[f"{prefix}_X{suffix}"],
            row[f"{prefix}_Y{suffix}"],
            row[f"{prefix}_Z{suffix}"],
        ],
        dtype=float,
    )


def build_displacement_comparison(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare baseline-to-follow-up photogrammetric and surveyed vectors."""

    displacement_results = []

    for target_point in TARGET_POINTS:
        point_results = results_df[
            results_df["point_label"] == target_point
        ].set_index("epoch")

        result_base = point_results.loc[BASE_EPOCH]
        result_followup = point_results.loc[FOLLOWUP_EPOCH]

        photo_base = xyz_from_row(result_base, "photo")
        photo_followup = xyz_from_row(result_followup, "photo")

        reference_base = xyz_from_row(result_base, "reference")
        reference_followup = xyz_from_row(result_followup, "reference")

        photo_displacement_mm = (
            photo_followup - photo_base
        ) * 1000.0

        reference_displacement_mm = (
            reference_followup - reference_base
        ) * 1000.0

        displacement_difference_mm = (
            photo_displacement_mm - reference_displacement_mm
        )

        displacement_results.append(
            {
                "point_label": target_point,
                f"observations_{BASE_EPOCH}": int(
                    result_base["observations"]
                ),
                f"observations_{FOLLOWUP_EPOCH}": int(
                    result_followup["observations"]
                ),
                "photo_delta_X_mm": photo_displacement_mm[0],
                "photo_delta_Y_mm": photo_displacement_mm[1],
                "photo_delta_Z_mm": photo_displacement_mm[2],
                "photo_displacement_3d_mm": np.linalg.norm(
                    photo_displacement_mm
                ),
                "geodetic_delta_X_mm": reference_displacement_mm[0],
                "geodetic_delta_Y_mm": reference_displacement_mm[1],
                "geodetic_delta_Z_mm": reference_displacement_mm[2],
                "geodetic_displacement_3d_mm": np.linalg.norm(
                    reference_displacement_mm
                ),
                "difference_X_mm": displacement_difference_mm[0],
                "difference_Y_mm": displacement_difference_mm[1],
                "difference_Z_mm": displacement_difference_mm[2],
                "vector_difference_3d_mm": np.linalg.norm(
                    displacement_difference_mm
                ),
            }
        )

    return pd.DataFrame(displacement_results)


def build_displacement_summary(
    displacement_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute group-level displacement comparison statistics."""

    difference_x = displacement_df["difference_X_mm"]
    difference_y = displacement_df["difference_Y_mm"]
    difference_z = displacement_df["difference_Z_mm"]
    vector_difference = displacement_df["vector_difference_3d_mm"]

    summary = {
        "mean_difference_X_mm": difference_x.mean(),
        "mean_difference_Y_mm": difference_y.mean(),
        "mean_difference_Z_mm": difference_z.mean(),
        "rmse_X_mm": np.sqrt(np.mean(difference_x**2)),
        "rmse_Y_mm": np.sqrt(np.mean(difference_y**2)),
        "rmse_Z_mm": np.sqrt(np.mean(difference_z**2)),
        "mean_vector_difference_mm": vector_difference.mean(),
        "median_vector_difference_mm": vector_difference.median(),
        "rmse_vector_difference_mm": np.sqrt(
            np.mean(vector_difference**2)
        ),
        "maximum_vector_difference_mm": vector_difference.max(),
    }

    return pd.DataFrame([summary])


# -----------------------------------------------------------------------------
# Leave-one-out diagnostics and influence analysis
# -----------------------------------------------------------------------------

def compute_loo_diagnostics(
    epoch_data_by_epoch: dict[str, dict],
) -> pd.DataFrame:
    """
    Compute leave-one-out reprojection errors for every observation.

    Potential high-residual observations are marked using a modified z-score.
    This is a relative screening heuristic, not a formal rejection test.
    """

    all_records = []

    for target_point in TARGET_POINTS:
        for epoch in EPOCHS:
            epoch_data = epoch_data_by_epoch[epoch]
            cameras = epoch_data["cameras"]

            observations = get_point_observations(
                epoch_data,
                target_point,
            )

            if len(observations) < 3:
                raise ValueError(
                    f"At least 3 observations are required for LOO analysis "
                    f"of {target_point}, {epoch}."
                )

            point_records = []

            for observation_index, observation in observations.iterrows():
                training_observations = observations.drop(
                    index=observation_index
                )

                xyz_loo = triangulate_from_observations(
                    training_observations,
                    cameras,
                )

                image_name = observation["image_name"]

                predicted_pixel = project_local_point_to_pixel(
                    xyz_loo,
                    cameras[image_name],
                )

                measured_pixel = np.array(
                    [observation["image_x"], observation["image_y"]],
                    dtype=float,
                )

                residual = measured_pixel - predicted_pixel
                error_px = float(np.linalg.norm(residual))

                point_records.append(
                    {
                        "point_label": target_point,
                        "epoch": epoch,
                        "image_name": image_name,
                        "loo_du_px": residual[0],
                        "loo_dv_px": residual[1],
                        "loo_error_px": error_px,
                    }
                )

            errors = np.array(
                [record["loo_error_px"] for record in point_records],
                dtype=float,
            )

            median_error = float(np.median(errors))
            mad_error = float(
                np.median(np.abs(errors - median_error))
            )

            for record in point_records:
                if mad_error > 1e-12:
                    modified_z = (
                        0.6745
                        * (record["loo_error_px"] - median_error)
                        / mad_error
                    )
                else:
                    modified_z = np.nan

                record["loo_median_error_px"] = median_error
                record["loo_mad_error_px"] = mad_error
                record["modified_z"] = modified_z
                record["potential_outlier"] = bool(
                    np.isfinite(modified_z)
                    and modified_z > LOO_MODIFIED_Z_THRESHOLD
                )

                all_records.append(record)

    return pd.DataFrame(all_records)


def compute_flagged_influence(
    flagged_loo: pd.DataFrame,
    epoch_data_by_epoch: dict[str, dict],
) -> pd.DataFrame:
    """Measure object-space influence of every LOO-flagged observation."""

    influence_records = []

    for _, flagged_row in flagged_loo.iterrows():
        point_label = flagged_row["point_label"]
        epoch = str(flagged_row["epoch"])
        excluded_image = flagged_row["image_name"]

        epoch_data = epoch_data_by_epoch[epoch]
        cameras = epoch_data["cameras"]

        observations = get_point_observations(
            epoch_data,
            point_label,
        )

        full_coordinate = triangulate_from_observations(
            observations,
            cameras,
        )

        reduced_observations = observations[
            observations["image_name"] != excluded_image
        ].copy()

        reduced_coordinate = triangulate_from_observations(
            reduced_observations,
            cameras,
        )

        coordinate_shift_mm = (
            reduced_coordinate - full_coordinate
        ) * 1000.0

        influence_records.append(
            {
                "point_label": point_label,
                "epoch": epoch,
                "image_name": excluded_image,
                "loo_error_px": flagged_row["loo_error_px"],
                "modified_z": flagged_row["modified_z"],
                "influence_X_mm": coordinate_shift_mm[0],
                "influence_Y_mm": coordinate_shift_mm[1],
                "influence_Z_mm": coordinate_shift_mm[2],
                "influence_3d_mm": np.linalg.norm(
                    coordinate_shift_mm
                ),
            }
        )

    return pd.DataFrame(influence_records)


# -----------------------------------------------------------------------------
# Robust triangulation
# -----------------------------------------------------------------------------

def build_robust_triangulation(
    results_df: pd.DataFrame,
    epoch_data_by_epoch: dict[str, dict],
    f_scale: float = HUBER_F_SCALE_PX,
) -> pd.DataFrame:
    """Run robust nonlinear image-space refinement for all point/epoch pairs."""

    robust_records = []

    baseline_indexed = results_df.set_index(["point_label", "epoch"])

    for target_point in TARGET_POINTS:
        for epoch in EPOCHS:
            epoch_data = epoch_data_by_epoch[epoch]
            cameras = epoch_data["cameras"]
            offset = epoch_data["offset"]

            observations = get_point_observations(
                epoch_data,
                target_point,
            )

            baseline_row = baseline_indexed.loc[
                (target_point, epoch)
            ]

            standard_xyz_local = np.array(
                [
                    baseline_row["photo_X_local_m"],
                    baseline_row["photo_Y_local_m"],
                    baseline_row["photo_Z_local_m"],
                ],
                dtype=float,
            )

            robust_xyz_local = robust_triangulate_from_observations(
                observations,
                cameras,
                standard_xyz_local,
                f_scale=f_scale,
            )

            standard_xyz_absolute = standard_xyz_local + offset
            robust_xyz_absolute = robust_xyz_local + offset

            robust_shift_mm = (
                robust_xyz_local - standard_xyz_local
            ) * 1000.0

            _, robust_reprojection_summary = (
                compute_reprojection_diagnostics(
                    robust_xyz_local,
                    observations,
                    cameras,
                )
            )

            robust_records.append(
                {
                    "point_label": target_point,
                    "epoch": epoch,
                    "observations": len(observations),
                    "standard_X_local_m": standard_xyz_local[0],
                    "standard_Y_local_m": standard_xyz_local[1],
                    "standard_Z_local_m": standard_xyz_local[2],
                    "robust_X_local_m": robust_xyz_local[0],
                    "robust_Y_local_m": robust_xyz_local[1],
                    "robust_Z_local_m": robust_xyz_local[2],
                    "standard_X_m": standard_xyz_absolute[0],
                    "standard_Y_m": standard_xyz_absolute[1],
                    "standard_Z_m": standard_xyz_absolute[2],
                    "robust_X_m": robust_xyz_absolute[0],
                    "robust_Y_m": robust_xyz_absolute[1],
                    "robust_Z_m": robust_xyz_absolute[2],
                    "robust_shift_X_mm": robust_shift_mm[0],
                    "robust_shift_Y_mm": robust_shift_mm[1],
                    "robust_shift_Z_mm": robust_shift_mm[2],
                    "robust_shift_3d_mm": np.linalg.norm(
                        robust_shift_mm
                    ),
                    "robust_reprojection_mean_px": (
                        robust_reprojection_summary[
                            "reprojection_mean_px"
                        ]
                    ),
                    "robust_reprojection_rmse_px": (
                        robust_reprojection_summary[
                            "reprojection_rmse_px"
                        ]
                    ),
                    "robust_reprojection_max_px": (
                        robust_reprojection_summary[
                            "reprojection_max_px"
                        ]
                    ),
                }
            )

    return pd.DataFrame(robust_records)


def build_robust_displacement_comparison(
    robust_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare robust photogrammetric displacement with surveyed displacement."""

    records = []

    robust_indexed = robust_df.set_index(["point_label", "epoch"])
    baseline_indexed = results_df.set_index(["point_label", "epoch"])

    for target_point in TARGET_POINTS:
        robust_base = robust_indexed.loc[
            (target_point, BASE_EPOCH)
        ]
        robust_followup = robust_indexed.loc[
            (target_point, FOLLOWUP_EPOCH)
        ]

        baseline_base = baseline_indexed.loc[
            (target_point, BASE_EPOCH)
        ]
        baseline_followup = baseline_indexed.loc[
            (target_point, FOLLOWUP_EPOCH)
        ]

        robust_base_absolute = np.array(
            [
                robust_base["robust_X_m"],
                robust_base["robust_Y_m"],
                robust_base["robust_Z_m"],
            ],
            dtype=float,
        )

        robust_followup_absolute = np.array(
            [
                robust_followup["robust_X_m"],
                robust_followup["robust_Y_m"],
                robust_followup["robust_Z_m"],
            ],
            dtype=float,
        )

        reference_base = xyz_from_row(
            baseline_base,
            "reference",
        )
        reference_followup = xyz_from_row(
            baseline_followup,
            "reference",
        )

        robust_displacement_mm = (
            robust_followup_absolute - robust_base_absolute
        ) * 1000.0

        geodetic_displacement_mm = (
            reference_followup - reference_base
        ) * 1000.0

        robust_difference_mm = (
            robust_displacement_mm - geodetic_displacement_mm
        )

        records.append(
            {
                "point_label": target_point,
                "robust_delta_X_mm": robust_displacement_mm[0],
                "robust_delta_Y_mm": robust_displacement_mm[1],
                "robust_delta_Z_mm": robust_displacement_mm[2],
                "robust_displacement_3d_mm": np.linalg.norm(
                    robust_displacement_mm
                ),
                "geodetic_delta_X_mm": geodetic_displacement_mm[0],
                "geodetic_delta_Y_mm": geodetic_displacement_mm[1],
                "geodetic_delta_Z_mm": geodetic_displacement_mm[2],
                "geodetic_displacement_3d_mm": np.linalg.norm(
                    geodetic_displacement_mm
                ),
                "difference_X_mm": robust_difference_mm[0],
                "difference_Y_mm": robust_difference_mm[1],
                "difference_Z_mm": robust_difference_mm[2],
                "vector_difference_3d_mm": np.linalg.norm(
                    robust_difference_mm
                ),
            }
        )

    return pd.DataFrame(records)


def build_robust_summary(
    robust_displacement_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute group-level statistics for robust displacement comparison."""

    dx = robust_displacement_df["difference_X_mm"]
    dy = robust_displacement_df["difference_Y_mm"]
    dz = robust_displacement_df["difference_Z_mm"]
    vector = robust_displacement_df["vector_difference_3d_mm"]

    return pd.DataFrame(
        [
            {
                "mean_difference_X_mm": dx.mean(),
                "mean_difference_Y_mm": dy.mean(),
                "mean_difference_Z_mm": dz.mean(),
                "rmse_X_mm": np.sqrt(np.mean(dx**2)),
                "rmse_Y_mm": np.sqrt(np.mean(dy**2)),
                "rmse_Z_mm": np.sqrt(np.mean(dz**2)),
                "mean_vector_difference_mm": vector.mean(),
                "median_vector_difference_mm": vector.median(),
                "rmse_vector_difference_mm": np.sqrt(
                    np.mean(vector**2)
                ),
                "maximum_vector_difference_mm": vector.max(),
            }
        ]
    )


def build_huber_scale_sensitivity(
    results_df: pd.DataFrame,
    displacement_df: pd.DataFrame,
    epoch_data_by_epoch: dict[str, dict],
) -> pd.DataFrame:
    """Evaluate displacement agreement over several Huber f_scale values."""

    records = []

    for f_scale in HUBER_SENSITIVITY_SCALES_PX:
        robust_epoch_df = build_robust_triangulation(
            results_df,
            epoch_data_by_epoch,
            f_scale=f_scale,
        )

        robust_displacement_df = build_robust_displacement_comparison(
            robust_epoch_df,
            results_df,
        )

        vector_differences = robust_displacement_df[
            "vector_difference_3d_mm"
        ].to_numpy(dtype=float)

        records.append(
            {
                "f_scale_px": f_scale,
                "mean_3d_discrepancy_mm": vector_differences.mean(),
                "rmse_3d_discrepancy_mm": np.sqrt(
                    np.mean(vector_differences**2)
                ),
                "maximum_3d_discrepancy_mm": vector_differences.max(),
            }
        )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Jackknife internal precision
# -----------------------------------------------------------------------------

def build_jackknife_precision(
    epoch_data_by_epoch: dict[str, dict],
    f_scale: float = HUBER_F_SCALE_PX,
) -> pd.DataFrame:
    """
    Estimate internal coordinate stability by leave-one-observation-out jackknife.

    The covariance captures sensitivity to image observations. It does not
    include all camera-parameter or reference-frame uncertainty sources.
    """

    records = []

    for target_point in TARGET_POINTS:
        for epoch in EPOCHS:
            epoch_data = epoch_data_by_epoch[epoch]
            observations = get_point_observations(
                epoch_data,
                target_point,
            )

            if len(observations) < 3:
                raise ValueError(
                    f"At least 3 observations are required for jackknife "
                    f"analysis of {target_point}, {epoch}."
                )

            jackknife_coordinates = []

            for observation_index in observations.index:
                reduced_observations = observations.drop(
                    index=observation_index
                )

                initial_xyz = triangulate_from_observations(
                    reduced_observations,
                    epoch_data["cameras"],
                )

                robust_xyz = robust_triangulate_from_observations(
                    reduced_observations,
                    epoch_data["cameras"],
                    initial_xyz,
                    f_scale=f_scale,
                )

                robust_absolute = (
                    robust_xyz + epoch_data["offset"]
                )

                jackknife_coordinates.append(
                    robust_absolute
                )

            jackknife_coordinates = np.asarray(
                jackknife_coordinates,
                dtype=float,
            )

            n_observations = len(jackknife_coordinates)
            jackknife_mean = jackknife_coordinates.mean(axis=0)

            centered = jackknife_coordinates - jackknife_mean

            jackknife_covariance_m2 = (
                (n_observations - 1)
                / n_observations
                * (centered.T @ centered)
            )

            jackknife_covariance_mm2 = (
                jackknife_covariance_m2 * 1_000_000.0
            )

            jackknife_se_mm = np.sqrt(
                np.diag(jackknife_covariance_mm2)
            )

            records.append(
                {
                    "point_label": target_point,
                    "epoch": epoch,
                    "observations": n_observations,
                    "jackknife_se_X_mm": jackknife_se_mm[0],
                    "jackknife_se_Y_mm": jackknife_se_mm[1],
                    "jackknife_se_Z_mm": jackknife_se_mm[2],
                    "jackknife_3d_scale_mm": np.linalg.norm(
                        jackknife_se_mm
                    ),
                    "cov_XX_mm2": jackknife_covariance_mm2[0, 0],
                    "cov_XY_mm2": jackknife_covariance_mm2[0, 1],
                    "cov_XZ_mm2": jackknife_covariance_mm2[0, 2],
                    "cov_YY_mm2": jackknife_covariance_mm2[1, 1],
                    "cov_YZ_mm2": jackknife_covariance_mm2[1, 2],
                    "cov_ZZ_mm2": jackknife_covariance_mm2[2, 2],
                }
            )

    return pd.DataFrame(records)


def covariance_matrix_from_row(row: pd.Series) -> np.ndarray:
    """Reconstruct a symmetric 3x3 covariance matrix from a result row."""

    return np.array(
        [
            [
                row["cov_XX_mm2"],
                row["cov_XY_mm2"],
                row["cov_XZ_mm2"],
            ],
            [
                row["cov_XY_mm2"],
                row["cov_YY_mm2"],
                row["cov_YZ_mm2"],
            ],
            [
                row["cov_XZ_mm2"],
                row["cov_YZ_mm2"],
                row["cov_ZZ_mm2"],
            ],
        ],
        dtype=float,
    )


def build_displacement_precision(
    jackknife_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Propagate jackknife coordinate precision to epoch-to-epoch displacement.

    This assumes the two epoch reconstruction errors are independent.
    """

    records = []
    indexed = jackknife_df.set_index(["point_label", "epoch"])

    for target_point in TARGET_POINTS:
        base = indexed.loc[(target_point, BASE_EPOCH)]
        followup = indexed.loc[(target_point, FOLLOWUP_EPOCH)]

        se_delta_x_mm = np.hypot(
            base["jackknife_se_X_mm"],
            followup["jackknife_se_X_mm"],
        )

        se_delta_y_mm = np.hypot(
            base["jackknife_se_Y_mm"],
            followup["jackknife_se_Y_mm"],
        )

        se_delta_z_mm = np.hypot(
            base["jackknife_se_Z_mm"],
            followup["jackknife_se_Z_mm"],
        )

        records.append(
            {
                "point_label": target_point,
                "jackknife_SE_delta_X_mm": se_delta_x_mm,
                "jackknife_SE_delta_Y_mm": se_delta_y_mm,
                "jackknife_SE_delta_Z_mm": se_delta_z_mm,
                "jackknife_displacement_3d_scale_mm": np.linalg.norm(
                    [
                        se_delta_x_mm,
                        se_delta_y_mm,
                        se_delta_z_mm,
                    ]
                ),
            }
        )

    return pd.DataFrame(records)


def build_precision_vs_discrepancy(
    robust_displacement_df: pd.DataFrame,
    displacement_precision_df: pd.DataFrame,
) -> pd.DataFrame:
    """Place internal displacement precision beside external discrepancy."""

    result = (
        robust_displacement_df[
            [
                "point_label",
                "vector_difference_3d_mm",
            ]
        ]
        .merge(
            displacement_precision_df,
            on="point_label",
            how="inner",
        )
    )

    result["discrepancy_to_internal_scale_ratio"] = (
        result["vector_difference_3d_mm"]
        / result["jackknife_displacement_3d_scale_mm"]
    )

    return result


# -----------------------------------------------------------------------------
# Full-covariance uncertainty analysis
# -----------------------------------------------------------------------------

def displacement_difference_vector(
    row: pd.Series,
) -> np.ndarray:
    """Return photogrammetric-minus-geodetic displacement difference vector."""

    return np.array(
        [
            row["difference_X_mm"],
            row["difference_Y_mm"],
            row["difference_Z_mm"],
        ],
        dtype=float,
    )


def photogrammetric_displacement_covariance(
    target_point: str,
    jackknife_df: pd.DataFrame,
) -> np.ndarray:
    """Combine the two epoch jackknife covariance matrices."""

    indexed = jackknife_df.set_index(["point_label", "epoch"])

    covariance_base = covariance_matrix_from_row(
        indexed.loc[(target_point, BASE_EPOCH)]
    )
    covariance_followup = covariance_matrix_from_row(
        indexed.loc[(target_point, FOLLOWUP_EPOCH)]
    )

    return covariance_base + covariance_followup


def mahalanobis_squared(
    vector: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Return v^T C^-1 v for a positive-definite covariance matrix."""

    try:
        solved = np.linalg.solve(covariance, vector)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Covariance matrix is singular; cannot compute "
            "the full-covariance statistic."
        ) from exc

    return float(vector @ solved)


def build_full_covariance_screen(
    robust_displacement_df: pd.DataFrame,
    jackknife_df: pd.DataFrame,
    geodetic_sigma_epoch_mm: float = GEODETIC_SIGMA_PER_EPOCH_MM,
) -> pd.DataFrame:
    """
    Compare method differences using the full photogrammetric covariance.

    Geodetic uncertainty is modeled as independent and isotropic per coordinate
    within each epoch, so the displacement contribution is 2*sigma^2*I.
    """

    if geodetic_sigma_epoch_mm < 0:
        raise ValueError("Geodetic sigma cannot be negative.")

    robust_indexed = robust_displacement_df.set_index("point_label")
    records = []

    for target_point in TARGET_POINTS:
        row = robust_indexed.loc[target_point]
        difference = displacement_difference_vector(row)

        photo_covariance = photogrammetric_displacement_covariance(
            target_point,
            jackknife_df,
        )

        geodetic_covariance = (
            2.0
            * geodetic_sigma_epoch_mm**2
            * np.eye(3)
        )

        combined_covariance = (
            photo_covariance + geodetic_covariance
        )

        statistic = mahalanobis_squared(
            difference,
            combined_covariance,
        )

        records.append(
            {
                "point_label": target_point,
                "geodetic_sigma_epoch_mm": geodetic_sigma_epoch_mm,
                "vector_difference_3d_mm": row[
                    "vector_difference_3d_mm"
                ],
                "chi2_full_covariance": statistic,
                "p_value_full_covariance": chi2.sf(
                    statistic,
                    df=CHI2_DF,
                ),
                "within_95pct_full_covariance": (
                    statistic <= CHI2_THRESHOLD_95
                ),
            }
        )

    return pd.DataFrame(records)


def build_full_covariance_sensitivity(
    robust_displacement_df: pd.DataFrame,
    jackknife_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the full-covariance screen over geodetic sigma scenarios."""

    tables = []

    for sigma_epoch_mm in GEODETIC_SIGMA_SCENARIOS_MM:
        scenario = build_full_covariance_screen(
            robust_displacement_df,
            jackknife_df,
            geodetic_sigma_epoch_mm=sigma_epoch_mm,
        ).rename(
            columns={
                "within_95pct_full_covariance":
                    "within_95pct_model",
            }
        )

        tables.append(
            scenario[
                [
                    "geodetic_sigma_epoch_mm",
                    "point_label",
                    "chi2_full_covariance",
                    "within_95pct_model",
                ]
            ]
        )

    sensitivity_df = pd.concat(
        tables,
        ignore_index=True,
    )

    summary_df = (
        sensitivity_df
        .groupby(
            "geodetic_sigma_epoch_mm"
        )["within_95pct_model"]
        .agg(
            points_within_95pct="sum",
            total_points="count",
        )
        .reset_index()
    )

    return sensitivity_df, summary_df


def find_critical_geodetic_sigma(
    difference: np.ndarray,
    photo_covariance: np.ndarray,
) -> tuple[float, float]:
    """
    Find the minimum per-epoch geodetic sigma that reaches the 95% boundary.

    Returns
    -------
    chi2_at_zero, critical_sigma_mm
    """

    def statistic_at_sigma(sigma_epoch_mm: float) -> float:
        combined_covariance = (
            photo_covariance
            + 2.0 * sigma_epoch_mm**2 * np.eye(3)
        )
        return mahalanobis_squared(
            difference,
            combined_covariance,
        )

    chi2_at_zero = statistic_at_sigma(0.0)

    if chi2_at_zero <= CHI2_THRESHOLD_95:
        return chi2_at_zero, 0.0

    def root_function(sigma_epoch_mm: float) -> float:
        return statistic_at_sigma(
            sigma_epoch_mm
        ) - CHI2_THRESHOLD_95

    upper = 1.0

    while root_function(upper) > 0:
        upper *= 2.0
        if upper > 1024.0:
            raise RuntimeError(
                "Could not bracket the critical geodetic sigma."
            )

    critical_sigma = brentq(
        root_function,
        0.0,
        upper,
    )

    return chi2_at_zero, float(critical_sigma)


def build_full_covariance_critical_thresholds(
    robust_displacement_df: pd.DataFrame,
    jackknife_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute full-covariance critical geodetic sigma for each point."""

    robust_indexed = robust_displacement_df.set_index("point_label")
    records = []

    for target_point in TARGET_POINTS:
        row = robust_indexed.loc[target_point]
        difference = displacement_difference_vector(row)

        photo_covariance = photogrammetric_displacement_covariance(
            target_point,
            jackknife_df,
        )

        chi2_at_zero, critical_sigma = find_critical_geodetic_sigma(
            difference,
            photo_covariance,
        )

        records.append(
            {
                "point_label": target_point,
                "chi2_at_zero_geodetic_sigma": chi2_at_zero,
                "critical_geodetic_sigma_epoch_mm": critical_sigma,
            }
        )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

def build_final_summary(
    displacement_df: pd.DataFrame,
    robust_displacement_df: pd.DataFrame,
    displacement_precision_df: pd.DataFrame,
    full_covariance_critical_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the main numerical result table used by plots and documentation."""

    final_summary_df = (
        displacement_df[
            [
                "point_label",
                "geodetic_delta_X_mm",
                "geodetic_delta_Y_mm",
                "geodetic_delta_Z_mm",
                "geodetic_displacement_3d_mm",
                "vector_difference_3d_mm",
            ]
        ]
        .rename(
            columns={
                "vector_difference_3d_mm":
                    "standard_vector_difference_3d_mm",
            }
        )
        .merge(
            robust_displacement_df[
                [
                    "point_label",
                    "robust_delta_X_mm",
                    "robust_delta_Y_mm",
                    "robust_delta_Z_mm",
                    "robust_displacement_3d_mm",
                    "vector_difference_3d_mm",
                ]
            ].rename(
                columns={
                    "vector_difference_3d_mm":
                        "robust_vector_difference_3d_mm",
                }
            ),
            on="point_label",
            how="inner",
        )
        .merge(
            displacement_precision_df[
                [
                    "point_label",
                    "jackknife_displacement_3d_scale_mm",
                ]
            ],
            on="point_label",
            how="inner",
        )
        .merge(
            full_covariance_critical_df[
                [
                    "point_label",
                    "critical_geodetic_sigma_epoch_mm",
                ]
            ],
            on="point_label",
            how="inner",
        )
    )

    final_summary_df[
        "robust_relative_discrepancy_percent"
    ] = (
        100.0
        * final_summary_df[
            "robust_vector_difference_3d_mm"
        ]
        / final_summary_df[
            "geodetic_displacement_3d_mm"
        ]
    )

    return final_summary_df[
        [
            "point_label",
            "geodetic_delta_X_mm",
            "geodetic_delta_Y_mm",
            "geodetic_delta_Z_mm",
            "geodetic_displacement_3d_mm",
            "robust_delta_X_mm",
            "robust_delta_Y_mm",
            "robust_delta_Z_mm",
            "robust_displacement_3d_mm",
            "standard_vector_difference_3d_mm",
            "robust_vector_difference_3d_mm",
            "robust_relative_discrepancy_percent",
            "jackknife_displacement_3d_scale_mm",
            "critical_geodetic_sigma_epoch_mm",
        ]
    ]


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def print_baseline_report(
    results_df: pd.DataFrame,
    reprojection_residuals_df: pd.DataFrame,
    displacement_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Print standard triangulation results and diagnostics."""

    print("\nLargest reprojection residuals:")

    for target_point in TARGET_POINTS:
        for epoch in EPOCHS:
            subset = reprojection_residuals_df[
                (
                    reprojection_residuals_df["point_label"]
                    == target_point
                )
                & (
                    reprojection_residuals_df["epoch"]
                    == epoch
                )
            ].sort_values(
                "error_px",
                ascending=False,
            )

            print(f"\n{target_point}, {epoch}:")
            for _, row in subset.head(3).iterrows():
                print(
                    f"{row['image_name']}: "
                    f"error = {row['error_px']:.3f} px, "
                    f"du = {row['residual_u_px']:.3f} px, "
                    f"dv = {row['residual_v_px']:.3f} px"
                )

    print("\nTriangulation results:")
    print(
        results_df[
            [
                "point_label",
                "epoch",
                "observations",
                "photo_X_m",
                "photo_Y_m",
                "photo_Z_m",
                "reference_X_m",
                "reference_Y_m",
                "reference_Z_m",
                "photo_minus_ref_3d_mm",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\nCamera-ray geometry:")
    print(
        results_df[
            [
                "point_label",
                "epoch",
                "observations",
                "geometry_lambda_min",
                "geometry_lambda_mid",
                "geometry_lambda_max",
                "geometry_condition_number",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nReprojection diagnostics:")
    print(
        results_df[
            [
                "point_label",
                "epoch",
                "observations",
                "reprojection_mean_px",
                "reprojection_rmse_px",
                "reprojection_max_px",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        f"\n{BASE_EPOCH} → {FOLLOWUP_EPOCH} "
        "displacement comparison:"
    )
    print(
        displacement_df[
            [
                "point_label",
                "photo_delta_X_mm",
                "geodetic_delta_X_mm",
                "photo_delta_Y_mm",
                "geodetic_delta_Y_mm",
                "photo_delta_Z_mm",
                "geodetic_delta_Z_mm",
                "vector_difference_3d_mm",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    summary = summary_df.iloc[0]

    print("\nGroup displacement comparison statistics:")
    print("\nMean component differences (photogrammetric - geodetic):")
    print(
        f"Mean ΔX difference: "
        f"{summary['mean_difference_X_mm']:.3f} mm"
    )
    print(
        f"Mean ΔY difference: "
        f"{summary['mean_difference_Y_mm']:.3f} mm"
    )
    print(
        f"Mean ΔZ difference: "
        f"{summary['mean_difference_Z_mm']:.3f} mm"
    )

    print("\nComponent RMSE:")
    print(f"RMSE X: {summary['rmse_X_mm']:.3f} mm")
    print(f"RMSE Y: {summary['rmse_Y_mm']:.3f} mm")
    print(f"RMSE Z: {summary['rmse_Z_mm']:.3f} mm")

    print("\n3D vector discrepancy statistics:")
    print(
        f"Mean: "
        f"{summary['mean_vector_difference_mm']:.3f} mm"
    )
    print(
        f"Median: "
        f"{summary['median_vector_difference_mm']:.3f} mm"
    )
    print(
        f"RMSE: "
        f"{summary['rmse_vector_difference_mm']:.3f} mm"
    )
    print(
        f"Maximum: "
        f"{summary['maximum_vector_difference_mm']:.3f} mm"
    )


def print_loo_report(
    loo_df: pd.DataFrame,
    flagged_influence_df: pd.DataFrame,
) -> None:
    """Print LOO potential outliers and object-space influence."""

    flagged_loo = loo_df[
        loo_df["potential_outlier"]
    ].sort_values(
        "modified_z",
        ascending=False,
    )

    print("\nPotential LOO outliers:")

    if flagged_loo.empty:
        print("No observations flagged.")
    else:
        print(
            flagged_loo[
                [
                    "point_label",
                    "epoch",
                    "image_name",
                    "loo_error_px",
                    "loo_median_error_px",
                    "loo_mad_error_px",
                    "modified_z",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )

    print("\nFlagged-observation influence analysis:")

    if flagged_influence_df.empty:
        print("No flagged observations to analyze.")
    else:
        print(
            flagged_influence_df[
                [
                    "point_label",
                    "epoch",
                    "image_name",
                    "loo_error_px",
                    "modified_z",
                    "influence_X_mm",
                    "influence_Y_mm",
                    "influence_Z_mm",
                    "influence_3d_mm",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )


def print_robust_report(
    robust_df: pd.DataFrame,
    robust_displacement_df: pd.DataFrame,
    robust_summary_df: pd.DataFrame,
) -> None:
    """Print standard-vs-robust changes and robust displacement comparison."""

    print("\nStandard vs robust triangulation:")
    print(
        robust_df[
            [
                "point_label",
                "epoch",
                "robust_shift_X_mm",
                "robust_shift_Y_mm",
                "robust_shift_Z_mm",
                "robust_shift_3d_mm",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nRobust displacement comparison:")
    print(
        robust_displacement_df[
            [
                "point_label",
                "robust_delta_X_mm",
                "geodetic_delta_X_mm",
                "robust_delta_Y_mm",
                "geodetic_delta_Y_mm",
                "robust_delta_Z_mm",
                "geodetic_delta_Z_mm",
                "vector_difference_3d_mm",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    summary = robust_summary_df.iloc[0]

    print("\nRobust group statistics:")
    print(
        f"Mean 3D vector discrepancy: "
        f"{summary['mean_vector_difference_mm']:.3f} mm"
    )
    print(
        f"3D vector RMSE: "
        f"{summary['rmse_vector_difference_mm']:.3f} mm"
    )


def print_extended_report(
    huber_sensitivity_df: pd.DataFrame,
    jackknife_df: pd.DataFrame,
    displacement_precision_df: pd.DataFrame,
    precision_vs_discrepancy_df: pd.DataFrame,
    full_covariance_screen_df: pd.DataFrame,
    full_covariance_sensitivity_df: pd.DataFrame,
    full_covariance_sensitivity_summary_df: pd.DataFrame,
    full_covariance_critical_df: pd.DataFrame,
    final_summary_df: pd.DataFrame,
) -> None:
    """Print robust-sensitivity, precision, and uncertainty results."""

    print("\nHuber-scale sensitivity:")
    print(
        huber_sensitivity_df
        .round(3)
        .to_string(index=False)
    )

    print("\nRobust jackknife coordinate precision:")
    print(
        jackknife_df
        .round(3)
        .to_string(index=False)
    )

    print("\nJackknife displacement precision:")
    print(
        displacement_precision_df
        .round(3)
        .to_string(index=False)
    )

    print("\nInternal precision vs external discrepancy:")
    print(
        precision_vs_discrepancy_df[
            [
                "point_label",
                "jackknife_displacement_3d_scale_mm",
                "vector_difference_3d_mm",
                "discrepancy_to_internal_scale_ratio",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nFull-covariance uncertainty screen:")
    print(
        full_covariance_screen_df[
            [
                "point_label",
                "vector_difference_3d_mm",
                "chi2_full_covariance",
                "p_value_full_covariance",
                "within_95pct_full_covariance",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print(
        f"\n95% chi-square threshold "
        f"(df={CHI2_DF}): "
        f"{CHI2_THRESHOLD_95:.3f}"
    )

    print(
        "\nFull-covariance geodetic "
        "uncertainty sensitivity:"
    )
    print(
        full_covariance_sensitivity_df
        .round(3)
        .to_string(index=False)
    )

    print("\nFull-covariance sensitivity summary:")
    print(
        full_covariance_sensitivity_summary_df
        .to_string(index=False)
    )

    print(
        "\nFull-covariance critical "
        "geodetic uncertainty thresholds:"
    )
    print(
        full_covariance_critical_df
        .round(3)
        .to_string(index=False)
    )

    print("\nFinal deformation-analysis summary:")
    print(
        final_summary_df
        .round(3)
        .to_string(index=False)
    )


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_outputs(
    outputs: dict[str, tuple[Path, pd.DataFrame]],
) -> None:
    """Save named DataFrames and print their destinations."""

    print("\nSaved output files:")

    for label, (path, dataframe) in outputs.items():
        dataframe.to_csv(path, index=False)
        print(f"{label}: {path}")


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------

def main() -> None:
    """Run the complete deformation-analysis pipeline."""

    # Load every epoch exactly once.
    epoch_data_by_epoch = {
        epoch: load_epoch_data(epoch)
        for epoch in EPOCHS
    }

    # 1) Standard multi-ray triangulation and baseline comparison.
    results_df, reprojection_residuals_df = (
        run_baseline_triangulation(
            epoch_data_by_epoch
        )
    )
    displacement_df = build_displacement_comparison(
        results_df
    )
    summary_df = build_displacement_summary(
        displacement_df
    )

    # 2) Leave-one-out image-space screening.
    loo_df = compute_loo_diagnostics(
        epoch_data_by_epoch
    )
    flagged_loo = loo_df[
        loo_df["potential_outlier"]
    ].sort_values(
        "modified_z",
        ascending=False,
    )
    flagged_influence_df = compute_flagged_influence(
        flagged_loo,
        epoch_data_by_epoch,
    )

    # 3) Robust nonlinear image-space refinement.
    robust_df = build_robust_triangulation(
        results_df,
        epoch_data_by_epoch,
        f_scale=HUBER_F_SCALE_PX,
    )
    robust_displacement_df = (
        build_robust_displacement_comparison(
            robust_df,
            results_df,
        )
    )
    robust_summary_df = build_robust_summary(
        robust_displacement_df
    )

    # 4) Huber-scale sensitivity.
    huber_sensitivity_df = build_huber_scale_sensitivity(
        results_df,
        displacement_df,
        epoch_data_by_epoch,
    )

    # 5) Internal photogrammetric precision.
    jackknife_df = build_jackknife_precision(
        epoch_data_by_epoch,
        f_scale=HUBER_F_SCALE_PX,
    )
    displacement_precision_df = build_displacement_precision(
        jackknife_df
    )
    precision_vs_discrepancy_df = (
        build_precision_vs_discrepancy(
            robust_displacement_df,
            displacement_precision_df,
        )
    )

    # 6) Full-covariance uncertainty comparison and sensitivity.
    full_covariance_screen_df = build_full_covariance_screen(
        robust_displacement_df,
        jackknife_df,
        geodetic_sigma_epoch_mm=GEODETIC_SIGMA_PER_EPOCH_MM,
    )

    (
        full_covariance_sensitivity_df,
        full_covariance_sensitivity_summary_df,
    ) = build_full_covariance_sensitivity(
        robust_displacement_df,
        jackknife_df,
    )

    full_covariance_critical_df = (
        build_full_covariance_critical_thresholds(
            robust_displacement_df,
            jackknife_df,
        )
    )

    # 7) Main final results table.
    final_summary_df = build_final_summary(
        displacement_df,
        robust_displacement_df,
        displacement_precision_df,
        full_covariance_critical_df,
    )

    # Reports.
    print_baseline_report(
        results_df,
        reprojection_residuals_df,
        displacement_df,
        summary_df,
    )
    print_loo_report(
        loo_df,
        flagged_influence_df,
    )
    print_robust_report(
        robust_df,
        robust_displacement_df,
        robust_summary_df,
    )
    print_extended_report(
        huber_sensitivity_df,
        jackknife_df,
        displacement_precision_df,
        precision_vs_discrepancy_df,
        full_covariance_screen_df,
        full_covariance_sensitivity_df,
        full_covariance_sensitivity_summary_df,
        full_covariance_critical_df,
        final_summary_df,
    )

    # Persist all useful tables. Intermediate files are reproducible and can
    # remain ignored by Git; the final summary is intentionally versioned.
    outputs = {
        "epoch_results": (
            PROCESSED_DIR / "triangulation_all_points_epochs.csv",
            results_df,
        ),
        "reprojection_residuals": (
            PROCESSED_DIR / "reprojection_residuals_all_points.csv",
            reprojection_residuals_df,
        ),
        "displacement_comparison": (
            PROCESSED_DIR / "displacement_comparison_all_points.csv",
            displacement_df,
        ),
        "displacement_summary": (
            PROCESSED_DIR / "displacement_comparison_summary.csv",
            summary_df,
        ),
        "loo_diagnostics": (
            PROCESSED_DIR / "loo_reprojection_diagnostics.csv",
            loo_df,
        ),
        "flagged_influence": (
            PROCESSED_DIR / "flagged_observation_influence.csv",
            flagged_influence_df,
        ),
        "robust_epoch_results": (
            PROCESSED_DIR / "robust_triangulation_all_points_epochs.csv",
            robust_df,
        ),
        "robust_displacement_comparison": (
            PROCESSED_DIR / "robust_displacement_comparison_all_points.csv",
            robust_displacement_df,
        ),
        "robust_displacement_summary": (
            PROCESSED_DIR / "robust_displacement_comparison_summary.csv",
            robust_summary_df,
        ),
        "huber_scale_sensitivity": (
            PROCESSED_DIR / "huber_scale_sensitivity.csv",
            huber_sensitivity_df,
        ),
        "jackknife_precision": (
            PROCESSED_DIR / "robust_jackknife_precision.csv",
            jackknife_df,
        ),
        "jackknife_displacement_precision": (
            PROCESSED_DIR / "robust_jackknife_displacement_precision.csv",
            displacement_precision_df,
        ),
        "precision_vs_discrepancy": (
            PROCESSED_DIR / "internal_precision_vs_external_discrepancy.csv",
            precision_vs_discrepancy_df,
        ),
        "full_covariance_uncertainty_screen": (
            PROCESSED_DIR / "full_covariance_uncertainty_screen.csv",
            full_covariance_screen_df,
        ),
        "full_covariance_sensitivity": (
            PROCESSED_DIR / "full_covariance_uncertainty_sensitivity.csv",
            full_covariance_sensitivity_df,
        ),
        "full_covariance_sensitivity_summary": (
            PROCESSED_DIR / "full_covariance_uncertainty_sensitivity_summary.csv",
            full_covariance_sensitivity_summary_df,
        ),
        "full_covariance_critical_thresholds": (
            PROCESSED_DIR / "full_covariance_critical_geodetic_uncertainty.csv",
            full_covariance_critical_df,
        ),
        "final_summary": (
            PROCESSED_DIR / "final_deformation_analysis_summary.csv",
            final_summary_df,
        ),
    }

    save_outputs(outputs)


if __name__ == "__main__":
    main()
