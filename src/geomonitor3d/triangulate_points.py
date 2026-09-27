from pathlib import Path
from scipy.stats import chi2
import numpy as np
import pandas as pd
from scipy.optimize import root, least_squares, brentq

# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = ["2018", "2024"]
TARGET_POINTS = ["R1", "R2", "R3", "R5", "R6", "R7"]

# Robust-estimation / diagnostic settings.
HUBER_F_SCALE_PX = 1.0
LOO_MODIFIED_Z_THRESHOLD = 3.5


# -----------------------------------------------------------------------------
# File/path helpers
# -----------------------------------------------------------------------------


def get_epoch_paths(epoch):
    """Return all input paths for one epoch."""

    param_dir = DATA_DIR / f"{epoch}_param"

    return {
        "param_dir": param_dir,
        "camera": param_dir / f"HV40_{epoch}_calibrated_camera_parameters.txt",
        "image_points": param_dir / f"HV40_{epoch}_image_points.txt",
        "object_points": param_dir / f"HV40_{epoch}_object_points.txt",
        "offset": param_dir / f"HV40_{epoch}_offset.xyz",
    }


# -----------------------------------------------------------------------------
# Input readers
# -----------------------------------------------------------------------------


def load_camera_parameters(file_path):
    """
    Read a Pix4D calibrated-camera-parameter file.

    Returns
    -------
    dict
        Dictionary keyed by image stem, e.g. ``DSC02929``.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    cameras = {}
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


def load_image_points(file_path):
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
        image_points["image_x"], errors="raise"
    )
    image_points["image_y"] = pd.to_numeric(
        image_points["image_y"], errors="raise"
    )

    return image_points


def load_object_points(file_path):
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

    return object_points


def load_offset(file_path):
    """Read the 3D epoch offset used by the local camera-coordinate system."""

    offset = np.asarray(np.loadtxt(file_path), dtype=float).reshape(-1)

    if offset.size != 3:
        raise ValueError(
            f"Expected exactly 3 offset values in {file_path}, "
            f"found {offset.size}."
        )

    return offset


def load_epoch_data(epoch):
    """Load all inputs required for one epoch once."""

    paths = get_epoch_paths(epoch)

    return {
        "epoch": epoch,
        "paths": paths,
        "cameras": load_camera_parameters(paths["camera"]),
        "image_points": load_image_points(paths["image_points"]),
        "object_points": load_object_points(paths["object_points"]),
        "offset": load_offset(paths["offset"]),
    }


# -----------------------------------------------------------------------------
# Camera model
# -----------------------------------------------------------------------------


def distort_normalized(x, y, radial, tangential):
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

    return x_distorted, y_distorted


def undistort_pixel(u, v, K, radial, tangential):
    """
    Convert a measured distorted pixel to undistorted normalized coordinates.
    """

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x_distorted = (u - cx) / fx
    y_distorted = (v - cy) / fy

    def equations(unknowns):
        x, y = unknowns
        predicted_x, predicted_y = distort_normalized(
            x,
            y,
            radial,
            tangential,
        )
        return [
            predicted_x - x_distorted,
            predicted_y - y_distorted,
        ]

    solution = root(
        equations,
        [x_distorted, y_distorted],
    )

    if not solution.success:
        raise RuntimeError(
            "Distortion inversion did not converge for "
            f"pixel ({u}, {v})."
        )

    return solution.x


def project_local_point_to_pixel(xyz_local, camera):
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
# Triangulation and diagnostics
# -----------------------------------------------------------------------------


def validate_observations(observations, cameras, point_label=None, epoch=None):
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


def build_object_space_rays(observations, cameras):
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
        direction_object /= np.linalg.norm(direction_object)

        ray_origins.append(camera["camera_position"])
        ray_directions.append(direction_object)

    return (
        np.asarray(ray_origins, dtype=float),
        np.asarray(ray_directions, dtype=float),
    )


def solve_multi_ray_intersection(ray_origins, ray_directions):
    """
    Solve the least-squares closest point to multiple 3D viewing rays.

    Returns the local 3D point plus geometry diagnostics.
    """

    identity = np.eye(3)
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)

    for camera_position, direction in zip(ray_origins, ray_directions):
        projection = identity - np.outer(direction, direction)
        A += projection
        b += projection @ camera_position

    normalized_A = A / len(ray_origins)
    geometry_eigenvalues = np.linalg.eigvalsh(normalized_A)

    lambda_min = geometry_eigenvalues[0]
    lambda_max = geometry_eigenvalues[-1]

    if lambda_min <= 1e-12:
        raise np.linalg.LinAlgError(
            "Triangulation geometry is singular or nearly singular."
        )

    geometry_condition_number = lambda_max / lambda_min

    xyz_local = np.linalg.solve(A, b)

    geometry = {
        "geometry_lambda_min": geometry_eigenvalues[0],
        "geometry_lambda_mid": geometry_eigenvalues[1],
        "geometry_lambda_max": geometry_eigenvalues[2],
        "geometry_condition_number": geometry_condition_number,
    }

    return xyz_local, geometry


def triangulate_from_observations(
    observations,
    cameras,
    return_geometry=False,
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
    xyz_local,
    observations,
    cameras,
):
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
        error_px = np.linalg.norm(residual)

        records.append(
            {
                "image_name": image_name,
                "residual_u_px": residual[0],
                "residual_v_px": residual[1],
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
    observations,
    cameras,
    initial_xyz,
    f_scale=1.0,
):
    """
    Refine a local 3D point by robustly minimizing pixel reprojection residuals.

    The surveyed/reference coordinate is not used in this estimation.
    """

    validate_observations(observations, cameras)

    def residual_function(xyz_local):
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
        initial_xyz,
        loss="huber",
        f_scale=f_scale,
    )

    if not result.success:
        raise RuntimeError(
            "Robust triangulation failed to converge: "
            f"{result.message}"
        )

    return result.x


# -----------------------------------------------------------------------------
# Epoch-level baseline solution
# -----------------------------------------------------------------------------


def triangulate_epoch(epoch, target_point, epoch_data):
    """
    Triangulate one target point for one epoch and compare it to the
    surveyed/reference coordinate.
    """

    cameras = epoch_data["cameras"]
    image_points = epoch_data["image_points"]
    object_points = epoch_data["object_points"]
    offset = epoch_data["offset"]

    observations = image_points[
        image_points["point_label"] == target_point
    ].copy()

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

    reference = object_points[
        object_points["GCP_label"] == target_point
    ]

    if len(reference) != 1:
        raise ValueError(
            f"Expected exactly one surveyed/reference coordinate for "
            f"{target_point} in {epoch}; found {len(reference)}."
        )

    xyz_reference = np.array(
        [
            reference.iloc[0]["GCP_measured_X"],
            reference.iloc[0]["GCP_measured_Y"],
            reference.iloc[0]["GCP_measured_Z"],
        ],
        dtype=float,
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


def run_baseline_triangulation(epoch_data_by_epoch):
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


# -----------------------------------------------------------------------------
# Displacement comparison
# -----------------------------------------------------------------------------


def build_displacement_comparison(results_df):
    """Compare 2018->2024 photogrammetric and surveyed displacement vectors."""

    displacement_results = []

    for target_point in TARGET_POINTS:
        point_results = results_df[
            results_df["point_label"] == target_point
        ].set_index("epoch")

        result_2018 = point_results.loc["2018"]
        result_2024 = point_results.loc["2024"]

        photo_2018 = np.array(
            [
                result_2018["photo_X_m"],
                result_2018["photo_Y_m"],
                result_2018["photo_Z_m"],
            ],
            dtype=float,
        )

        photo_2024 = np.array(
            [
                result_2024["photo_X_m"],
                result_2024["photo_Y_m"],
                result_2024["photo_Z_m"],
            ],
            dtype=float,
        )

        reference_2018 = np.array(
            [
                result_2018["reference_X_m"],
                result_2018["reference_Y_m"],
                result_2018["reference_Z_m"],
            ],
            dtype=float,
        )

        reference_2024 = np.array(
            [
                result_2024["reference_X_m"],
                result_2024["reference_Y_m"],
                result_2024["reference_Z_m"],
            ],
            dtype=float,
        )

        photo_displacement_mm = (
            photo_2024 - photo_2018
        ) * 1000.0

        reference_displacement_mm = (
            reference_2024 - reference_2018
        ) * 1000.0

        displacement_difference_mm = (
            photo_displacement_mm - reference_displacement_mm
        )

        displacement_results.append(
            {
                "point_label": target_point,
                "observations_2018": int(result_2018["observations"]),
                "observations_2024": int(result_2024["observations"]),
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


def build_displacement_summary(displacement_df):
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


def compute_loo_diagnostics(epoch_data_by_epoch):
    """
    Compute leave-one-out reprojection errors for every observation and mark
    high relative residuals using a modified z-score.
    """

    all_records = []

    for target_point in TARGET_POINTS:
        for epoch in EPOCHS:
            epoch_data = epoch_data_by_epoch[epoch]
            cameras = epoch_data["cameras"]
            image_points = epoch_data["image_points"]

            observations = image_points[
                image_points["point_label"] == target_point
            ].copy()

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
                error_px = np.linalg.norm(residual)

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


def compute_flagged_influence(flagged_loo, epoch_data_by_epoch):
    """Measure object-space influence of every LOO-flagged observation."""

    influence_records = []

    for _, flagged_row in flagged_loo.iterrows():
        point_label = flagged_row["point_label"]
        epoch = str(flagged_row["epoch"])
        excluded_image = flagged_row["image_name"]

        epoch_data = epoch_data_by_epoch[epoch]
        cameras = epoch_data["cameras"]
        image_points = epoch_data["image_points"]

        observations = image_points[
            image_points["point_label"] == point_label
        ].copy()

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
# Robust triangulation and robust displacement comparison
# -----------------------------------------------------------------------------


def build_robust_triangulation(results_df, epoch_data_by_epoch):
    """Run robust nonlinear image-space refinement for all point/epoch pairs."""

    robust_records = []

    for target_point in TARGET_POINTS:
        for epoch in EPOCHS:
            epoch_data = epoch_data_by_epoch[epoch]
            cameras = epoch_data["cameras"]
            image_points = epoch_data["image_points"]
            offset = epoch_data["offset"]

            observations = image_points[
                image_points["point_label"] == target_point
            ].copy()

            baseline_row = results_df[
                (results_df["point_label"] == target_point)
                & (results_df["epoch"] == epoch)
            ].iloc[0]

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
            )

            standard_xyz_absolute = standard_xyz_local + offset
            robust_xyz_absolute = robust_xyz_local + offset

            robust_shift_mm = (
                robust_xyz_local - standard_xyz_local
            ) * 1000.0

            robust_residual_df, robust_reprojection_summary = (
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


def build_robust_displacement_comparison(robust_df, results_df):
    """Compare robust photogrammetric displacement with surveyed displacement."""

    records = []

    for target_point in TARGET_POINTS:
        point_robust = robust_df[
            robust_df["point_label"] == target_point
        ].set_index("epoch")

        point_baseline = results_df[
            results_df["point_label"] == target_point
        ].set_index("epoch")

        robust_2018_absolute = np.array(
            [
                point_robust.loc["2018", "robust_X_m"],
                point_robust.loc["2018", "robust_Y_m"],
                point_robust.loc["2018", "robust_Z_m"],
            ],
            dtype=float,
        )

        robust_2024_absolute = np.array(
            [
                point_robust.loc["2024", "robust_X_m"],
                point_robust.loc["2024", "robust_Y_m"],
                point_robust.loc["2024", "robust_Z_m"],
            ],
            dtype=float,
        )

        reference_2018 = np.array(
            [
                point_baseline.loc["2018", "reference_X_m"],
                point_baseline.loc["2018", "reference_Y_m"],
                point_baseline.loc["2018", "reference_Z_m"],
            ],
            dtype=float,
        )

        reference_2024 = np.array(
            [
                point_baseline.loc["2024", "reference_X_m"],
                point_baseline.loc["2024", "reference_Y_m"],
                point_baseline.loc["2024", "reference_Z_m"],
            ],
            dtype=float,
        )

        robust_displacement_mm = (
            robust_2024_absolute - robust_2018_absolute
        ) * 1000.0

        geodetic_displacement_mm = (
            reference_2024 - reference_2018
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


def build_robust_summary(robust_displacement_df):
    """Compute group-level statistics for the robust displacement comparison."""

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


# -----------------------------------------------------------------------------
# Output / reporting helpers
# -----------------------------------------------------------------------------


def print_baseline_report(
    results_df,
    reprojection_residuals_df,
    displacement_df,
    summary_df,
):
    """Print the standard triangulation results and diagnostics."""

    print("\nLargest reprojection residuals:")

    for target_point in TARGET_POINTS:
        for epoch in EPOCHS:
            subset = reprojection_residuals_df[
                (reprojection_residuals_df["point_label"] == target_point)
                & (reprojection_residuals_df["epoch"] == epoch)
            ].sort_values("error_px", ascending=False)

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

    print("\n2018 → 2024 displacement comparison:")
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
    print(f"Mean ΔX difference: {summary['mean_difference_X_mm']:.3f} mm")
    print(f"Mean ΔY difference: {summary['mean_difference_Y_mm']:.3f} mm")
    print(f"Mean ΔZ difference: {summary['mean_difference_Z_mm']:.3f} mm")

    print("\nComponent RMSE:")
    print(f"RMSE X: {summary['rmse_X_mm']:.3f} mm")
    print(f"RMSE Y: {summary['rmse_Y_mm']:.3f} mm")
    print(f"RMSE Z: {summary['rmse_Z_mm']:.3f} mm")

    print("\n3D vector discrepancy statistics:")
    print(f"Mean: {summary['mean_vector_difference_mm']:.3f} mm")
    print(f"Median: {summary['median_vector_difference_mm']:.3f} mm")
    print(f"RMSE: {summary['rmse_vector_difference_mm']:.3f} mm")
    print(f"Maximum: {summary['maximum_vector_difference_mm']:.3f} mm")


def print_loo_report(loo_df, flagged_influence_df):
    """Print LOO outliers and their object-space influence."""

    flagged_loo = loo_df[loo_df["potential_outlier"]].sort_values(
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


def print_robust_report(robust_df, robust_displacement_df, robust_summary_df):
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


def save_outputs(
    results_df,
    reprojection_residuals_df,
    displacement_df,
    summary_df,
    loo_df,
    flagged_influence_df,
    robust_df,
    robust_displacement_df,
    robust_summary_df,
):
    """Save all processed outputs to data/processed."""

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
    }

    print("\nSaved output files:")

    for label, (path, dataframe) in outputs.items():
        dataframe.to_csv(path, index=False)
        print(f"{label}: {path}")


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------


def main():
    """Run the complete triangulation, diagnostic, and robust-analysis pipeline."""

    # Load each epoch only once.
    epoch_data_by_epoch = {
        epoch: load_epoch_data(epoch)
        for epoch in EPOCHS
    }

    # 1) Standard multi-ray triangulation.
    results_df, reprojection_residuals_df = run_baseline_triangulation(
        epoch_data_by_epoch
    )

    # 2) Standard 2018->2024 displacement comparison.
    displacement_df = build_displacement_comparison(results_df)
    summary_df = build_displacement_summary(displacement_df)

    print_baseline_report(
        results_df,
        reprojection_residuals_df,
        displacement_df,
        summary_df,
    )

    # 3) Leave-one-out image-space screening.
    loo_df = compute_loo_diagnostics(epoch_data_by_epoch)
    flagged_loo = loo_df[loo_df["potential_outlier"]].sort_values(
        "modified_z",
        ascending=False,
    )
    flagged_influence_df = compute_flagged_influence(
        flagged_loo,
        epoch_data_by_epoch,
    )

    print_loo_report(
        loo_df,
        flagged_influence_df,
    )

    # 4) Robust nonlinear refinement using Huber loss.
    robust_df = build_robust_triangulation(
        results_df,
        epoch_data_by_epoch,
    )

    # 5) Robust 2018->2024 displacement comparison.
    # Robust absolute coordinates were already created by adding each epoch's
    # own offset inside build_robust_triangulation(), so there are no undefined
    # offset_2018 / offset_2024 variables here.
    robust_displacement_df = build_robust_displacement_comparison(
        robust_df,
        results_df,
    )
    robust_summary_df = build_robust_summary(
        robust_displacement_df
    )

    print_robust_report(
        robust_df,
        robust_displacement_df,
        robust_summary_df,
    )

    # 6) Save all tables.
    save_outputs(
        results_df,
        reprojection_residuals_df,
        displacement_df,
        summary_df,
        loo_df,
        flagged_influence_df,
        robust_df,
        robust_displacement_df,
        robust_summary_df,
    )
    
    
    huber_scales = [
        0.5,
        1.0,
        2.0,
    ]
    
    
    sensitivity_records = []
    
    
    for f_scale in huber_scales:
    
        robust_epoch_records = []
    
    
        for target_point in TARGET_POINTS:
    
            for epoch in EPOCHS:
    
                epoch_data = load_epoch_data(
                    epoch
                )
    
                observations = (
                    epoch_data["image_points"][
                        epoch_data[
                            "image_points"
                        ]["point_label"]
                        == target_point
                    ]
                    .copy()
                )
    
    
                standard_xyz = (
                    triangulate_from_observations(
                        observations,
                        epoch_data["cameras"],
                    )
                )
    
    
                robust_xyz = (
                    robust_triangulate_from_observations(
                        observations,
                        epoch_data["cameras"],
                        standard_xyz,
                        f_scale=f_scale,
                    )
                )
    
    
                robust_absolute = (
                    robust_xyz
                    + epoch_data["offset"]
                )
    
    
                robust_epoch_records.append(
                    {
                        "point_label":
                            target_point,
    
                        "epoch":
                            epoch,
    
                        "X":
                            robust_absolute[0],
    
                        "Y":
                            robust_absolute[1],
    
                        "Z":
                            robust_absolute[2],
                    }
                )
    
    
        robust_epoch_df = pd.DataFrame(
            robust_epoch_records
        )
    
    
        vector_differences = []
    
    
        for target_point in TARGET_POINTS:
    
            point_robust = (
                robust_epoch_df[
                    robust_epoch_df["point_label"]
                    == target_point
                ]
                .set_index("epoch")
            )
    
    
            robust_2018 = np.array(
                [
                    point_robust.loc["2018", "X"],
                    point_robust.loc["2018", "Y"],
                    point_robust.loc["2018", "Z"],
                ]
            )
    
    
            robust_2024 = np.array(
                [
                    point_robust.loc["2024", "X"],
                    point_robust.loc["2024", "Y"],
                    point_robust.loc["2024", "Z"],
                ]
            )
    
    
            robust_displacement_mm = (
                (
                    robust_2024
                    - robust_2018
                )
                * 1000
            )
    
    
            reference_row = (
                displacement_df[
                    displacement_df["point_label"]
                    == target_point
                ]
                .iloc[0]
            )
    
    
            geodetic_displacement_mm = np.array(
                [
                    reference_row[
                        "geodetic_delta_X_mm"
                    ],
                    reference_row[
                        "geodetic_delta_Y_mm"
                    ],
                    reference_row[
                        "geodetic_delta_Z_mm"
                    ],
                ]
            )
    
    
            difference = (
                robust_displacement_mm
                - geodetic_displacement_mm
            )
    
    
            vector_differences.append(
                np.linalg.norm(
                    difference
                )
            )
    
    
        vector_differences = np.array(
            vector_differences
        )
    
    
        sensitivity_records.append(
            {
                "f_scale_px":
                    f_scale,
    
                "mean_3d_discrepancy_mm":
                    vector_differences.mean(),
    
                "rmse_3d_discrepancy_mm":
                    np.sqrt(
                        np.mean(
                            vector_differences**2
                        )
                    ),
    
                "maximum_3d_discrepancy_mm":
                    vector_differences.max(),
            }
        )
    
    
    sensitivity_df = pd.DataFrame(
        sensitivity_records
    )
    
    
    print(
        "\nHuber-scale sensitivity:"
    )
    
    print(
        sensitivity_df
        .round(3)
        .to_string(
            index=False
        )
    )
    
    jackknife_records = []

    
    for target_point in TARGET_POINTS:
    
        for epoch in EPOCHS:
    
            epoch_data = load_epoch_data(
                epoch
            )
    
            observations = (
                epoch_data["image_points"][
                    epoch_data[
                        "image_points"
                    ]["point_label"]
                    == target_point
                ]
                .copy()
            )
    
    
            jackknife_coordinates = []
    
    
            for observation_index in observations.index:
    
                reduced_observations = (
                    observations.drop(
                        index=observation_index
                    )
                )
    
    
                initial_xyz = (
                    triangulate_from_observations(
                        reduced_observations,
                        epoch_data["cameras"],
                    )
                )
    
    
                robust_xyz = (
                    robust_triangulate_from_observations(
                        reduced_observations,
                        epoch_data["cameras"],
                        initial_xyz,
                        f_scale=1.0,
                    )
                )
    
    
                robust_absolute = (
                    robust_xyz
                    + epoch_data["offset"]
                )
    
    
                jackknife_coordinates.append(
                    robust_absolute
                )
    
    
            jackknife_coordinates = np.array(
                jackknife_coordinates
            )
    
    
            n_observations = len(
                jackknife_coordinates
            )
    
    
            jackknife_mean = (
                jackknife_coordinates.mean(
                    axis=0
                )
            )
            
            centered_coordinates = (
                jackknife_coordinates
                - jackknife_mean
            )
            
            
            jackknife_covariance_m2 = (
                (n_observations - 1)
                / n_observations
                * (
                    centered_coordinates.T
                    @ centered_coordinates
                )
            )
            
            
            jackknife_covariance_mm2 = (
                jackknife_covariance_m2
                * 1_000_000
            )
    
    
            squared_deviations = (
                (
                    jackknife_coordinates
                    - jackknife_mean
                )
                ** 2
            )
    
    
            jackknife_variance = (
                (n_observations - 1)
                / n_observations
                * squared_deviations.sum(
                    axis=0
                )
            )
    
    
            jackknife_se_m = np.sqrt(
                jackknife_variance
            )
    
    
            jackknife_se_mm = (
                jackknife_se_m
                * 1000
            )
    
    
            jackknife_3d_scale_mm = (
                np.linalg.norm(
                    jackknife_se_mm
                )
            )
    
    
            jackknife_records.append(
                {
                    "point_label":
                        target_point,
    
                    "epoch":
                        epoch,
    
                    "observations":
                        n_observations,
    
                    "jackknife_se_X_mm":
                        jackknife_se_mm[0],
    
                    "jackknife_se_Y_mm":
                        jackknife_se_mm[1],
    
                    "jackknife_se_Z_mm":
                        jackknife_se_mm[2],
    
                    "jackknife_3d_scale_mm":
                        jackknife_3d_scale_mm,
                        
                    "cov_XX_mm2":
                        jackknife_covariance_mm2[0, 0],
                    
                    "cov_XY_mm2":
                        jackknife_covariance_mm2[0, 1],
                    
                    "cov_XZ_mm2":
                        jackknife_covariance_mm2[0, 2],
                    
                    "cov_YY_mm2":
                        jackknife_covariance_mm2[1, 1],
                    
                    "cov_YZ_mm2":
                        jackknife_covariance_mm2[1, 2],
                    
                    "cov_ZZ_mm2":
                        jackknife_covariance_mm2[2, 2],
                }
            )
    
    
    jackknife_df = pd.DataFrame(
        jackknife_records
    )
    
    
    print(
        "\nRobust jackknife coordinate precision:"
    )
    
    
    print(
        jackknife_df
        .round(3)
        .to_string(
            index=False
        )
    )
    
    
    JACKKNIFE_OUTPUT_FILE = (
        PROCESSED_DIR
        / "robust_jackknife_precision.csv"
    )
    
    
    jackknife_df.to_csv(
        JACKKNIFE_OUTPUT_FILE,
        index=False,
    )
    
    
    print(
        f"\nSaved jackknife precision to:\n"
        f"{JACKKNIFE_OUTPUT_FILE}"
    )
    
    displacement_precision_records = []
    
    
    for target_point in TARGET_POINTS:
    
        point_jackknife = (
            jackknife_df[
                jackknife_df["point_label"]
                == target_point
            ]
            .set_index("epoch")
        )
    
    
        jackknife_2018 = point_jackknife.loc[
            "2018"
        ]
    
        jackknife_2024 = point_jackknife.loc[
            "2024"
        ]
    
    
        se_delta_x_mm = np.sqrt(
            jackknife_2018[
                "jackknife_se_X_mm"
            ] ** 2
            +
            jackknife_2024[
                "jackknife_se_X_mm"
            ] ** 2
        )
    
    
        se_delta_y_mm = np.sqrt(
            jackknife_2018[
                "jackknife_se_Y_mm"
            ] ** 2
            +
            jackknife_2024[
                "jackknife_se_Y_mm"
            ] ** 2
        )
    
    
        se_delta_z_mm = np.sqrt(
            jackknife_2018[
                "jackknife_se_Z_mm"
            ] ** 2
            +
            jackknife_2024[
                "jackknife_se_Z_mm"
            ] ** 2
        )
    
    
        displacement_3d_scale_mm = np.linalg.norm(
            [
                se_delta_x_mm,
                se_delta_y_mm,
                se_delta_z_mm,
            ]
        )
    
    
        displacement_precision_records.append(
            {
                "point_label":
                    target_point,
    
                "jackknife_SE_delta_X_mm":
                    se_delta_x_mm,
    
                "jackknife_SE_delta_Y_mm":
                    se_delta_y_mm,
    
                "jackknife_SE_delta_Z_mm":
                    se_delta_z_mm,
    
                "jackknife_displacement_3d_scale_mm":
                    displacement_3d_scale_mm,
            }
        )
    
    
    displacement_precision_df = pd.DataFrame(
        displacement_precision_records
    )
    
    
    print(
        "\nJackknife displacement precision:"
    )
    
    
    print(
        displacement_precision_df
        .round(3)
        .to_string(
            index=False
        )
    )
    
    
    DISPLACEMENT_PRECISION_FILE = (
        PROCESSED_DIR
        / "robust_jackknife_displacement_precision.csv"
    )
    
    
    displacement_precision_df.to_csv(
        DISPLACEMENT_PRECISION_FILE,
        index=False,
    )
    
    
    print(
        f"\nSaved displacement precision to:\n"
        f"{DISPLACEMENT_PRECISION_FILE}"
    )
    
    precision_vs_discrepancy_df = (
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
    
    
    precision_vs_discrepancy_df[
        "discrepancy_to_internal_scale_ratio"
    ] = (
        precision_vs_discrepancy_df[
            "vector_difference_3d_mm"
        ]
        /
        precision_vs_discrepancy_df[
            "jackknife_displacement_3d_scale_mm"
        ]
    )
    
    
    print(
        "\nInternal precision vs external discrepancy:"
    )
    
    
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
        .to_string(
            index=False
        )
    )
    
    
    PRECISION_COMPARISON_FILE = (
        PROCESSED_DIR
        / "internal_precision_vs_external_discrepancy.csv"
    )
    
    
    precision_vs_discrepancy_df.to_csv(
        PRECISION_COMPARISON_FILE,
        index=False,
    )
    
    
    print(
        f"\nSaved precision comparison to:\n"
        f"{PRECISION_COMPARISON_FILE}"
    )
    
    GEODETIC_SIGMA_PER_EPOCH_MM = 4.0

    
    geodetic_displacement_sigma_mm = np.sqrt(
        2
    ) * GEODETIC_SIGMA_PER_EPOCH_MM
    
    
    combined_uncertainty_df = (
        robust_displacement_df[
            [
                "point_label",
                "difference_X_mm",
                "difference_Y_mm",
                "difference_Z_mm",
                "vector_difference_3d_mm",
            ]
        ]
        .merge(
            displacement_precision_df,
            on="point_label",
            how="inner",
        )
    )
    
    
    combined_uncertainty_df[
        "combined_sigma_X_mm"
    ] = np.sqrt(
        combined_uncertainty_df[
            "jackknife_SE_delta_X_mm"
        ] ** 2
        +
        geodetic_displacement_sigma_mm**2
    )
    
    
    combined_uncertainty_df[
        "combined_sigma_Y_mm"
    ] = np.sqrt(
        combined_uncertainty_df[
            "jackknife_SE_delta_Y_mm"
        ] ** 2
        +
        geodetic_displacement_sigma_mm**2
    )
    
    
    combined_uncertainty_df[
        "combined_sigma_Z_mm"
    ] = np.sqrt(
        combined_uncertainty_df[
            "jackknife_SE_delta_Z_mm"
        ] ** 2
        +
        geodetic_displacement_sigma_mm**2
    )
        
            
    combined_uncertainty_df[
        "chi2_statistic"
    ] = (
        (
            combined_uncertainty_df[
                "difference_X_mm"
            ]
            /
            combined_uncertainty_df[
                "combined_sigma_X_mm"
            ]
        ) ** 2
        +
        (
            combined_uncertainty_df[
                "difference_Y_mm"
            ]
            /
            combined_uncertainty_df[
                "combined_sigma_Y_mm"
            ]
        ) ** 2
        +
        (
            combined_uncertainty_df[
                "difference_Z_mm"
            ]
            /
            combined_uncertainty_df[
                "combined_sigma_Z_mm"
            ]
        ) ** 2
    )
    
    
    combined_uncertainty_df[
        "p_value"
    ] = chi2.sf(
        combined_uncertainty_df[
            "chi2_statistic"
        ],
        df=3,
    )
    
    
    chi2_threshold_95 = chi2.ppf(
        0.95,
        df=3,
    )
    
    
    combined_uncertainty_df[
        "within_95pct_model"
    ] = (
        combined_uncertainty_df[
            "chi2_statistic"
        ]
        <= chi2_threshold_95
    )
         
    print(
        "\nCombined photogrammetric + geodetic uncertainty screen:"
    )
    
    print(
        combined_uncertainty_df[
            [
                "point_label",
                "vector_difference_3d_mm",
                "combined_sigma_X_mm",
                "combined_sigma_Y_mm",
                "combined_sigma_Z_mm",
                "chi2_statistic",
                "p_value",
                "within_95pct_model",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )
    
    
    print(
        f"\n95% chi-square threshold "
        f"(df=3): "
        f"{chi2_threshold_95:.3f}"
    )
    
    COMBINED_UNCERTAINTY_FILE = (
        PROCESSED_DIR
        / "combined_uncertainty_screen.csv"
    )
    
    
    combined_uncertainty_df.to_csv(
        COMBINED_UNCERTAINTY_FILE,
        index=False,
    )
    
    
    GEODETIC_SIGMA_SCENARIOS_MM = [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    
    
    uncertainty_sensitivity_records = []
    
    
    for sigma_epoch_mm in GEODETIC_SIGMA_SCENARIOS_MM:
    
        sigma_displacement_mm = (
            np.sqrt(2)
            * sigma_epoch_mm
        )
    
    
        scenario_df = (
            robust_displacement_df[
                [
                    "point_label",
                    "difference_X_mm",
                    "difference_Y_mm",
                    "difference_Z_mm",
                    "vector_difference_3d_mm",
                ]
            ]
            .merge(
                displacement_precision_df,
                on="point_label",
                how="inner",
            )
        )
    
    
        sigma_x = np.sqrt(
            scenario_df[
                "jackknife_SE_delta_X_mm"
            ] ** 2
            +
            sigma_displacement_mm**2
        )
    
    
        sigma_y = np.sqrt(
            scenario_df[
                "jackknife_SE_delta_Y_mm"
            ] ** 2
            +
            sigma_displacement_mm**2
        )
    
    
        sigma_z = np.sqrt(
            scenario_df[
                "jackknife_SE_delta_Z_mm"
            ] ** 2
            +
            sigma_displacement_mm**2
        )
    
    
        chi2_values = (
            (
                scenario_df[
                    "difference_X_mm"
                ]
                / sigma_x
            ) ** 2
            +
            (
                scenario_df[
                    "difference_Y_mm"
                ]
                / sigma_y
            ) ** 2
            +
            (
                scenario_df[
                    "difference_Z_mm"
                ]
                / sigma_z
            ) ** 2
        )
    
    
        for index, row in scenario_df.iterrows():
    
            chi2_value = (
                chi2_values.iloc[index]
            )
    
    
            uncertainty_sensitivity_records.append(
                {
                    "geodetic_sigma_epoch_mm":
                        sigma_epoch_mm,
    
                    "point_label":
                        row["point_label"],
    
                    "chi2_statistic":
                        chi2_value,
    
                    "within_95pct_model":
                        chi2_value
                        <= chi2_threshold_95,
                }
            )
    
    
    uncertainty_sensitivity_df = pd.DataFrame(
        uncertainty_sensitivity_records
    )
    
    
    print(
        "\nGeodetic uncertainty sensitivity:"
    )
    
    
    print(
        uncertainty_sensitivity_df
        .round(3)
        .to_string(
            index=False
        )
    )
    
    
    scenario_summary_df = (
        uncertainty_sensitivity_df
        .groupby(
            "geodetic_sigma_epoch_mm"
        )["within_95pct_model"]
        .agg(
            points_within_95pct="sum",
            total_points="count",
        )
        .reset_index()
    )
    
    
    print(
        "\nSensitivity summary:"
    )
    
    
    print(
        scenario_summary_df
        .to_string(
            index=False
        )
    )
    
    critical_sigma_records = []
    
    
    critical_sigma_input_df = (
        robust_displacement_df[
            [
                "point_label",
                "difference_X_mm",
                "difference_Y_mm",
                "difference_Z_mm",
            ]
        ]
        .merge(
            displacement_precision_df,
            on="point_label",
            how="inner",
        )
    )
    
    
    for _, row in critical_sigma_input_df.iterrows():
    
        def chi2_at_geodetic_sigma(
            sigma_epoch_mm,
        ):
            sigma_displacement_mm = (
                np.sqrt(2)
                * sigma_epoch_mm
            )
    
    
            sigma_x = np.sqrt(
                row[
                    "jackknife_SE_delta_X_mm"
                ] ** 2
                +
                sigma_displacement_mm**2
            )
    
            sigma_y = np.sqrt(
                row[
                    "jackknife_SE_delta_Y_mm"
                ] ** 2
                +
                sigma_displacement_mm**2
            )
    
            sigma_z = np.sqrt(
                row[
                    "jackknife_SE_delta_Z_mm"
                ] ** 2
                +
                sigma_displacement_mm**2
            )
    
    
            return (
                (
                    row["difference_X_mm"]
                    / sigma_x
                ) ** 2
                +
                (
                    row["difference_Y_mm"]
                    / sigma_y
                ) ** 2
                +
                (
                    row["difference_Z_mm"]
                    / sigma_z
                ) ** 2
            )
    
    
        chi2_with_zero_geo_uncertainty = (
            chi2_at_geodetic_sigma(
                0.0
            )
        )
    
    
        if (
            chi2_with_zero_geo_uncertainty
            <= chi2_threshold_95
        ):
    
            critical_sigma_mm = 0.0
    
        else:
    
            def root_function(
                sigma_epoch_mm,
            ):
                return (
                    chi2_at_geodetic_sigma(
                        sigma_epoch_mm
                    )
                    - chi2_threshold_95
                )
    
    
            critical_sigma_mm = brentq(
                root_function,
                0.0,
                20.0,
            )
    
    
        critical_sigma_records.append(
            {
                "point_label":
                    row["point_label"],
    
                "chi2_if_geo_sigma_zero":
                    chi2_with_zero_geo_uncertainty,
    
                "critical_geodetic_sigma_epoch_mm":
                    critical_sigma_mm,
            }
        )
    
    
    critical_sigma_df = pd.DataFrame(
        critical_sigma_records
    )
    
    
    print(
        "\nCritical geodetic uncertainty thresholds:"
    )
    
    
    print(
        critical_sigma_df
        .round(3)
        .to_string(
            index=False
        )
    )
    
    
    CRITICAL_SIGMA_FILE = (
        PROCESSED_DIR
        / "critical_geodetic_uncertainty_thresholds.csv"
    )
    
    
    critical_sigma_df.to_csv(
        CRITICAL_SIGMA_FILE,
        index=False,
    )
    
    
    print(
        f"\nSaved critical thresholds to:\n"
        f"{CRITICAL_SIGMA_FILE}"
    )
    
    
    covariance_test_records = []
    
    
    GEODETIC_SIGMA_EPOCH_MM = 4.0
    
    
    for target_point in TARGET_POINTS:
    
        point_jackknife = (
            jackknife_df[
                jackknife_df["point_label"]
                == target_point
            ]
            .set_index("epoch")
        )
    
    
        def covariance_matrix_from_row(
            row,
        ):
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
    
    
        covariance_2018 = (
            covariance_matrix_from_row(
                point_jackknife.loc["2018"]
            )
        )
    
    
        covariance_2024 = (
            covariance_matrix_from_row(
                point_jackknife.loc["2024"]
            )
        )
    
    
        photogrammetric_displacement_covariance = (
            covariance_2018
            + covariance_2024
        )
    
    
        geodetic_displacement_covariance = (
            2
            * GEODETIC_SIGMA_EPOCH_MM**2
            * np.eye(3)
        )
    
    
        combined_covariance = (
            photogrammetric_displacement_covariance
            + geodetic_displacement_covariance
        )
    
    
        displacement_row = (
            robust_displacement_df[
                robust_displacement_df[
                    "point_label"
                ]
                == target_point
            ]
            .iloc[0]
        )
    
    
        difference_vector = np.array(
            [
                displacement_row[
                    "difference_X_mm"
                ],
                displacement_row[
                    "difference_Y_mm"
                ],
                displacement_row[
                    "difference_Z_mm"
                ],
            ]
        )
    
    
        chi2_full_covariance = (
            difference_vector
            @ np.linalg.solve(
                combined_covariance,
                difference_vector,
            )
        )
    
    
        p_value_full_covariance = (
            chi2.sf(
                chi2_full_covariance,
                df=3,
            )
        )
    
    
        covariance_test_records.append(
            {
                "point_label":
                    target_point,
    
                "chi2_full_covariance":
                    chi2_full_covariance,
    
                "p_value_full_covariance":
                    p_value_full_covariance,
    
                "within_95pct_full_covariance":
                    chi2_full_covariance
                    <= chi2_threshold_95,
            }
        )
    
    
    covariance_test_df = pd.DataFrame(
        covariance_test_records
    )
    
    
    print(
        "\nFull-covariance uncertainty screen:"
    )
    
    
    print(
        covariance_test_df
        .round(3)
        .to_string(
            index=False
        )
    )
    
    
    FULL_COVARIANCE_SIGMA_SCENARIOS_MM = [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    
    
    full_covariance_sensitivity_records = []
    
    
    for sigma_epoch_mm in (
        FULL_COVARIANCE_SIGMA_SCENARIOS_MM
    ):
    
        for target_point in TARGET_POINTS:
    
            point_jackknife = (
                jackknife_df[
                    jackknife_df["point_label"]
                    == target_point
                ]
                .set_index("epoch")
            )
    
    
            covariance_2018 = (
                covariance_matrix_from_row(
                    point_jackknife.loc["2018"]
                )
            )
    
            covariance_2024 = (
                covariance_matrix_from_row(
                    point_jackknife.loc["2024"]
                )
            )
    
    
            photogrammetric_covariance = (
                covariance_2018
                + covariance_2024
            )
    
    
            geodetic_covariance = (
                2
                * sigma_epoch_mm**2
                * np.eye(3)
            )
    
    
            combined_covariance = (
                photogrammetric_covariance
                + geodetic_covariance
            )
    
    
            displacement_row = (
                robust_displacement_df[
                    robust_displacement_df[
                        "point_label"
                    ]
                    == target_point
                ]
                .iloc[0]
            )
    
    
            difference_vector = np.array(
                [
                    displacement_row[
                        "difference_X_mm"
                    ],
                    displacement_row[
                        "difference_Y_mm"
                    ],
                    displacement_row[
                        "difference_Z_mm"
                    ],
                ]
            )
    
    
            chi2_value = (
                difference_vector
                @ np.linalg.solve(
                    combined_covariance,
                    difference_vector,
                )
            )
    
    
            full_covariance_sensitivity_records.append(
                {
                    "geodetic_sigma_epoch_mm":
                        sigma_epoch_mm,
    
                    "point_label":
                        target_point,
    
                    "chi2_full_covariance":
                        chi2_value,
    
                    "within_95pct_model":
                        chi2_value
                        <= chi2_threshold_95,
                }
            )
    
    
    full_covariance_sensitivity_df = pd.DataFrame(
        full_covariance_sensitivity_records
    )
    
    
    print(
        "\nFull-covariance geodetic "
        "uncertainty sensitivity:"
    )
    
    
    print(
        full_covariance_sensitivity_df
        .round(3)
        .to_string(
            index=False
        )
    )
    
    full_covariance_sensitivity_summary = (
        full_covariance_sensitivity_df
        .groupby(
            "geodetic_sigma_epoch_mm"
        )["within_95pct_model"]
        .agg(
            points_within_95pct="sum",
            total_points="count",
        )
        .reset_index()
    )
    
    
    print(
        "\nFull-covariance sensitivity summary:"
    )
    
    
    print(
        full_covariance_sensitivity_summary
        .to_string(
            index=False
        )
    )
    
    full_covariance_critical_records = []
    
    
    for target_point in TARGET_POINTS:
    
        point_jackknife = (
            jackknife_df[
                jackknife_df["point_label"]
                == target_point
            ]
            .set_index("epoch")
        )
    
    
        covariance_2018 = (
            covariance_matrix_from_row(
                point_jackknife.loc["2018"]
            )
        )
    
        covariance_2024 = (
            covariance_matrix_from_row(
                point_jackknife.loc["2024"]
            )
        )
    
    
        photogrammetric_covariance = (
            covariance_2018
            + covariance_2024
        )
    
    
        displacement_row = (
            robust_displacement_df[
                robust_displacement_df[
                    "point_label"
                ]
                == target_point
            ]
            .iloc[0]
        )
    
    
        difference_vector = np.array(
            [
                displacement_row[
                    "difference_X_mm"
                ],
                displacement_row[
                    "difference_Y_mm"
                ],
                displacement_row[
                    "difference_Z_mm"
                ],
            ]
        )
    
    
        def chi2_full_at_sigma(
            sigma_epoch_mm,
        ):
    
            combined_covariance = (
                photogrammetric_covariance
                +
                2
                * sigma_epoch_mm**2
                * np.eye(3)
            )
    
    
            return (
                difference_vector
                @ np.linalg.solve(
                    combined_covariance,
                    difference_vector,
                )
            )
    
    
        chi2_zero = (
            chi2_full_at_sigma(
                0.0
            )
        )
    
    
        if (
            chi2_zero
            <= chi2_threshold_95
        ):
    
            critical_sigma_mm = 0.0
    
        else:
    
            def root_function(
                sigma_epoch_mm,
            ):
                return (
                    chi2_full_at_sigma(
                        sigma_epoch_mm
                    )
                    - chi2_threshold_95
                )
    
    
            critical_sigma_mm = brentq(
                root_function,
                0.0,
                20.0,
            )
    
    
        full_covariance_critical_records.append(
            {
                "point_label":
                    target_point,
    
                "chi2_at_zero_geodetic_sigma":
                    chi2_zero,
    
                "critical_geodetic_sigma_epoch_mm":
                    critical_sigma_mm,
            }
        )
    
    
    full_covariance_critical_df = pd.DataFrame(
        full_covariance_critical_records
    )
    
    
    print(
        "\nFull-covariance critical "
        "geodetic uncertainty thresholds:"
    )
    
    
    print(
        full_covariance_critical_df
        .round(3)
        .to_string(
            index=False
        )
    )
    
    
    FULL_COVARIANCE_CRITICAL_FILE = (
        PROCESSED_DIR
        / "full_covariance_critical_geodetic_uncertainty.csv"
    )
    
    
    full_covariance_critical_df.to_csv(
        FULL_COVARIANCE_CRITICAL_FILE,
        index=False,
    )
    
    
    print(
        f"\nSaved full-covariance critical thresholds to:\n"
        f"{FULL_COVARIANCE_CRITICAL_FILE}"
    )
    
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
                    "vector_difference_3d_mm",
                ]
            ]
            .rename(
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
        "robust_displacement_3d_mm"
    ] = np.sqrt(
        final_summary_df[
            "robust_delta_X_mm"
        ] ** 2
        +
        final_summary_df[
            "robust_delta_Y_mm"
        ] ** 2
        +
        final_summary_df[
            "robust_delta_Z_mm"
        ] ** 2
    )
        
    final_summary_df[
        "robust_relative_discrepancy_percent"
    ] = (
        100
        * final_summary_df[
            "robust_vector_difference_3d_mm"
        ]
        / final_summary_df[
            "geodetic_displacement_3d_mm"
        ]
    )
         
    final_summary_df = final_summary_df[
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
    
    print(
        "\nFinal deformation-analysis summary:"
    )
    
    
    print(
        final_summary_df
        .round(3)
        .to_string(
            index=False
        )
    )
    
    FINAL_SUMMARY_FILE = (
        PROCESSED_DIR
        / "final_deformation_analysis_summary.csv"
    )
    
    
    final_summary_df.to_csv(
        FINAL_SUMMARY_FILE,
        index=False,
    )
    
    
    print(
        f"\nSaved final analysis summary to:\n"
        f"{FINAL_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
