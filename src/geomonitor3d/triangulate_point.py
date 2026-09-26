from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EPOCHS = [
    "2018",
    "2024",
]

TARGET_POINT = "R5"


def load_camera_parameters(file_path):
    """
    Read a Pix4D calibrated-camera-parameter file.

    Returns a dictionary keyed by image name.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    cameras = {}

    i = 0

    while i < len(lines):
        parts = lines[i].split()

        is_camera_line = (
            len(parts) == 3
            and parts[0]
            .lower()
            .endswith(
                (
                    ".jpg",
                    ".jpeg",
                )
            )
        )

        if not is_camera_line:
            i += 1
            continue

        image_name = Path(
            parts[0]
        ).stem

        image_width = int(
            parts[1]
        )

        image_height = int(
            parts[2]
        )

        K = np.array(
            [
                [
                    float(value)
                    for value
                    in lines[i + 1].split()
                ],
                [
                    float(value)
                    for value
                    in lines[i + 2].split()
                ],
                [
                    float(value)
                    for value
                    in lines[i + 3].split()
                ],
            ],
            dtype=float,
        )

        radial = np.array(
            [
                float(value)
                for value
                in lines[i + 4].split()
            ],
            dtype=float,
        )

        tangential = np.array(
            [
                float(value)
                for value
                in lines[i + 5].split()
            ],
            dtype=float,
        )

        camera_position = np.array(
            [
                float(value)
                for value
                in lines[i + 6].split()
            ],
            dtype=float,
        )

        rotation = np.array(
            [
                [
                    float(value)
                    for value
                    in lines[i + 7].split()
                ],
                [
                    float(value)
                    for value
                    in lines[i + 8].split()
                ],
                [
                    float(value)
                    for value
                    in lines[i + 9].split()
                ],
            ],
            dtype=float,
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

    return cameras


def load_image_points(file_path):
    """
    Read the headerless image-point file.
    """

    return pd.read_csv(
        file_path,
        header=None,
        names=[
            "image_name",
            "point_label",
            "image_x",
            "image_y",
        ],
    )


def distort_normalized(
    x,
    y,
    radial,
    tangential,
):
    """
    Apply radial and tangential lens distortion
    to normalized coordinates.
    """

    k1, k2, k3 = radial
    t1, t2 = tangential

    r_squared = (
        x**2
        + y**2
    )

    radial_factor = (
        1
        + k1 * r_squared
        + k2 * r_squared**2
        + k3 * r_squared**3
    )

    x_distorted = (
        x * radial_factor
        + 2 * t1 * x * y
        + t2
        * (
            r_squared
            + 2 * x**2
        )
    )

    y_distorted = (
        y * radial_factor
        + t1
        * (
            r_squared
            + 2 * y**2
        )
        + 2 * t2 * x * y
    )

    return (
        x_distorted,
        y_distorted,
    )


def undistort_pixel(
    u,
    v,
    K,
    radial,
    tangential,
):
    """
    Convert a measured distorted pixel
    into undistorted normalized coordinates.
    """

    fx = K[0, 0]
    fy = K[1, 1]

    cx = K[0, 2]
    cy = K[1, 2]

    x_distorted = (
        u - cx
    ) / fx

    y_distorted = (
        v - cy
    ) / fy

    def equations(unknowns):
        x, y = unknowns

        (
            predicted_x,
            predicted_y,
        ) = distort_normalized(
            x,
            y,
            radial,
            tangential,
        )

        return [
            predicted_x
            - x_distorted,

            predicted_y
            - y_distorted,
        ]

    solution = root(
        equations,
        [
            x_distorted,
            y_distorted,
        ],
    )

    if not solution.success:
        raise RuntimeError(
            "Distortion inversion "
            "did not converge."
        )

    return solution.x


def triangulate_epoch(
    epoch,
    target_point,
):
    """
    Triangulate one target for one epoch.

    Returns both the photogrammetric
    coordinate and surveyed reference.
    """

    param_dir = (
        DATA_DIR
        / f"{epoch}_param"
    )

    camera_file = (
        param_dir
        / f"HV40_{epoch}_calibrated_camera_parameters.txt"
    )

    image_points_file = (
        param_dir
        / f"HV40_{epoch}_image_points.txt"
    )

    object_points_file = (
        param_dir
        / f"HV40_{epoch}_object_points.txt"
    )

    offset_file = (
        param_dir
        / f"HV40_{epoch}_offset.xyz"
    )

    cameras = load_camera_parameters(
        camera_file
    )

    image_points = load_image_points(
        image_points_file
    )

    object_points = pd.read_csv(
        object_points_file
    )

    offset = np.loadtxt(
        offset_file
    )

    observations = image_points[
        image_points["point_label"]
        == target_point
    ].copy()

    if len(observations) < 2:
        raise ValueError(
            f"{target_point} has only "
            f"{len(observations)} observations "
            f"in {epoch}."
        )

    ray_origins = []
    ray_directions = []

    for _, observation in (
        observations.iterrows()
    ):
        image_name = observation[
            "image_name"
        ]

        if image_name not in cameras:
            raise KeyError(
                f"No camera parameters found "
                f"for image {image_name} "
                f"in epoch {epoch}."
            )

        camera = cameras[
            image_name
        ]

        (
            x_undistorted,
            y_undistorted,
        ) = undistort_pixel(
            observation["image_x"],
            observation["image_y"],
            camera["K"],
            camera["radial"],
            camera["tangential"],
        )

        direction_camera = np.array(
            [
                x_undistorted,
                y_undistorted,
                1.0,
            ]
        )

        direction_object = (
            camera["rotation"].T
            @ direction_camera
        )

        direction_object = (
            direction_object
            / np.linalg.norm(
                direction_object
            )
        )

        ray_origins.append(
            camera[
                "camera_position"
            ]
        )

        ray_directions.append(
            direction_object
        )

    ray_origins = np.array(
        ray_origins
    )

    ray_directions = np.array(
        ray_directions
    )

    identity = np.eye(
        3
    )

    A = np.zeros(
        (
            3,
            3,
        )
    )

    b = np.zeros(
        3
    )

    for (
        camera_position,
        direction,
    ) in zip(
        ray_origins,
        ray_directions,
    ):
        projection = (
            identity
            - np.outer(
                direction,
                direction,
            )
        )

        A += projection

        b += (
            projection
            @ camera_position
        )

    xyz_local_photo = (
        np.linalg.solve(
            A,
            b,
        )
    )

    xyz_absolute_photo = (
        xyz_local_photo
        + offset
    )

    reference = object_points[
        object_points["GCP_label"]
        == target_point
    ]

    if len(reference) != 1:
        raise ValueError(
            f"Expected exactly one "
            f"surveyed coordinate for "
            f"{target_point} in {epoch}."
        )

    xyz_reference = np.array(
        [
            reference.iloc[0][
                "GCP_measured_X"
            ],
            reference.iloc[0][
                "GCP_measured_Y"
            ],
            reference.iloc[0][
                "GCP_measured_Z"
            ],
        ],
        dtype=float,
    )

    coordinate_difference_mm = (
        (
            xyz_absolute_photo
            - xyz_reference
        )
        * 1000
    )

    coordinate_difference_3d_mm = (
        np.linalg.norm(
            coordinate_difference_mm
        )
    )

    return {
        "epoch": epoch,
        "observations": len(
            observations
        ),

        "photo_X_m":
            xyz_absolute_photo[0],

        "photo_Y_m":
            xyz_absolute_photo[1],

        "photo_Z_m":
            xyz_absolute_photo[2],

        "reference_X_m":
            xyz_reference[0],

        "reference_Y_m":
            xyz_reference[1],

        "reference_Z_m":
            xyz_reference[2],

        "photo_minus_ref_X_mm":
            coordinate_difference_mm[0],

        "photo_minus_ref_Y_mm":
            coordinate_difference_mm[1],

        "photo_minus_ref_Z_mm":
            coordinate_difference_mm[2],

        "photo_minus_ref_3d_mm":
            coordinate_difference_3d_mm,
    }


results = []

for epoch in EPOCHS:
    result = triangulate_epoch(
        epoch,
        TARGET_POINT,
    )

    results.append(
        result
    )


results_df = pd.DataFrame(
    results
)


print(
    f"\nTriangulation results "
    f"for {TARGET_POINT}:"
)

display_columns = [
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

print(
    results_df[
        display_columns
    ]
    .round(6)
    .to_string(
        index=False
    )
)


result_2018 = results_df[
    results_df["epoch"]
    == "2018"
].iloc[0]

result_2024 = results_df[
    results_df["epoch"]
    == "2024"
].iloc[0]


photo_2018 = np.array(
    [
        result_2018["photo_X_m"],
        result_2018["photo_Y_m"],
        result_2018["photo_Z_m"],
    ]
)

photo_2024 = np.array(
    [
        result_2024["photo_X_m"],
        result_2024["photo_Y_m"],
        result_2024["photo_Z_m"],
    ]
)


reference_2018 = np.array(
    [
        result_2018["reference_X_m"],
        result_2018["reference_Y_m"],
        result_2018["reference_Z_m"],
    ]
)

reference_2024 = np.array(
    [
        result_2024["reference_X_m"],
        result_2024["reference_Y_m"],
        result_2024["reference_Z_m"],
    ]
)


photo_displacement_mm = (
    (
        photo_2024
        - photo_2018
    )
    * 1000
)

reference_displacement_mm = (
    (
        reference_2024
        - reference_2018
    )
    * 1000
)


photo_displacement_3d_mm = (
    np.linalg.norm(
        photo_displacement_mm
    )
)

reference_displacement_3d_mm = (
    np.linalg.norm(
        reference_displacement_mm
    )
)


displacement_difference_mm = (
    photo_displacement_mm
    - reference_displacement_mm
)

displacement_difference_3d_mm = (
    np.linalg.norm(
        displacement_difference_mm
    )
)


print(
    "\nPhotogrammetric "
    "2018 → 2024 displacement:"
)

print(
    f"ΔX = "
    f"{photo_displacement_mm[0]:.3f} mm"
)

print(
    f"ΔY = "
    f"{photo_displacement_mm[1]:.3f} mm"
)

print(
    f"ΔZ = "
    f"{photo_displacement_mm[2]:.3f} mm"
)

print(
    f"3D = "
    f"{photo_displacement_3d_mm:.3f} mm"
)


print(
    "\nSurveyed/geodetic "
    "2018 → 2024 displacement:"
)

print(
    f"ΔX = "
    f"{reference_displacement_mm[0]:.3f} mm"
)

print(
    f"ΔY = "
    f"{reference_displacement_mm[1]:.3f} mm"
)

print(
    f"ΔZ = "
    f"{reference_displacement_mm[2]:.3f} mm"
)

print(
    f"3D = "
    f"{reference_displacement_3d_mm:.3f} mm"
)


print(
    "\nPhotogrammetric minus "
    "geodetic displacement:"
)

print(
    f"ΔX difference = "
    f"{displacement_difference_mm[0]:.3f} mm"
)

print(
    f"ΔY difference = "
    f"{displacement_difference_mm[1]:.3f} mm"
)

print(
    f"ΔZ difference = "
    f"{displacement_difference_mm[2]:.3f} mm"
)

print(
    f"3D vector difference = "
    f"{displacement_difference_3d_mm:.3f} mm"
)


EPOCH_OUTPUT_FILE = (
    PROCESSED_DIR
    / f"{TARGET_POINT}_triangulation_epochs.csv"
)

results_df.to_csv(
    EPOCH_OUTPUT_FILE,
    index=False,
)


displacement_df = pd.DataFrame(
    [
        {
            "method":
                "photogrammetric",

            "delta_X_mm":
                photo_displacement_mm[0],

            "delta_Y_mm":
                photo_displacement_mm[1],

            "delta_Z_mm":
                photo_displacement_mm[2],

            "displacement_3d_mm":
                photo_displacement_3d_mm,
        },
        {
            "method":
                "geodetic",

            "delta_X_mm":
                reference_displacement_mm[0],

            "delta_Y_mm":
                reference_displacement_mm[1],

            "delta_Z_mm":
                reference_displacement_mm[2],

            "displacement_3d_mm":
                reference_displacement_3d_mm,
        },
    ]
)


DISPLACEMENT_OUTPUT_FILE = (
    PROCESSED_DIR
    / f"{TARGET_POINT}_displacement_comparison.csv"
)

displacement_df.to_csv(
    DISPLACEMENT_OUTPUT_FILE,
    index=False,
)


print(
    f"\nSaved epoch results to:\n"
    f"{EPOCH_OUTPUT_FILE}"
)

print(
    f"\nSaved displacement comparison to:\n"
    f"{DISPLACEMENT_OUTPUT_FILE}"
)