from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAMERA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "2018_param"
    / "HV40_2018_calibrated_camera_parameters.txt"
)

IMAGE_POINTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "2018_param"
    / "HV40_2018_image_points.txt"
)

OBJECT_POINTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "2018_param"
    / "HV40_2018_object_points.txt"
)

OFFSET_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "2018_param"
    / "HV40_2018_offset.xyz"
)


TARGET_IMAGE = "DSC02929"
TARGET_POINT = "R5"


def load_camera_parameters(file_path):
    """
    Read the Pix4D calibrated camera-parameter file.

    Returns a dictionary where each image name contains:
    - image width and height
    - intrinsic matrix K
    - radial distortion
    - tangential distortion
    - camera position
    - rotation matrix
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
            ]
        )

        radial = np.array(
            [
                float(value)
                for value
                in lines[i + 4].split()
            ]
        )

        tangential = np.array(
            [
                float(value)
                for value
                in lines[i + 5].split()
            ]
        )

        camera_position = np.array(
            [
                float(value)
                for value
                in lines[i + 6].split()
            ]
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
            ]
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


cameras = load_camera_parameters(
    CAMERA_FILE
)

image_points = load_image_points(
    IMAGE_POINTS_FILE
)

object_points = pd.read_csv(
    OBJECT_POINTS_FILE
)

offset = np.loadtxt(
    OFFSET_FILE
)

camera = cameras[
    TARGET_IMAGE
]


observation = image_points[
    (
        image_points["image_name"]
        == TARGET_IMAGE
    )
    & (
        image_points["point_label"]
        == TARGET_POINT
    )
]


if len(observation) != 1:
    raise ValueError(
        f"Expected exactly one observation "
        f"for {TARGET_POINT} "
        f"in {TARGET_IMAGE}, "
        f"but found {len(observation)}."
    )


u_measured = (
    observation.iloc[0][
        "image_x"
    ]
)

v_measured = (
    observation.iloc[0][
        "image_y"
    ]
)


K = camera[
    "K"
]

radial = camera[
    "radial"
]

tangential = camera[
    "tangential"
]


fx = K[0, 0]
fy = K[1, 1]

cx = K[0, 2]
cy = K[1, 2]


k1, k2, k3 = radial

t1, t2 = tangential


x_distorted = (
    u_measured - cx
) / fx

y_distorted = (
    v_measured - cy
) / fy


def distort_normalized(
    x,
    y,
):
    """
    Apply radial and tangential distortion
    to normalized image coordinates.
    """

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

    x_d = (
        x * radial_factor
        + 2 * t1 * x * y
        + t2
        * (
            r_squared
            + 2 * x**2
        )
    )

    y_d = (
        y * radial_factor
        + t1
        * (
            r_squared
            + 2 * y**2
        )
        + 2 * t2 * x * y
    )

    return x_d, y_d


def distortion_equations(
    unknowns,
):
    """
    Equations used to recover the
    undistorted normalized coordinate.
    """

    (
        x_undistorted,
        y_undistorted,
    ) = unknowns

    (
        predicted_x,
        predicted_y,
    ) = distort_normalized(
        x_undistorted,
        y_undistorted,
    )

    return [
        predicted_x
        - x_distorted,

        predicted_y
        - y_distorted,
    ]


solution = root(
    distortion_equations,
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


(
    x_undistorted,
    y_undistorted,
) = solution.x


u_undistorted = (
    fx * x_undistorted
    + cx
)

v_undistorted = (
    fy * y_undistorted
    + cy
)


(
    x_roundtrip,
    y_roundtrip,
) = distort_normalized(
    x_undistorted,
    y_undistorted,
)


u_roundtrip = (
    fx * x_roundtrip
    + cx
)

v_roundtrip = (
    fy * y_roundtrip
    + cy
)


roundtrip_error = np.hypot(
    u_roundtrip
    - u_measured,

    v_roundtrip
    - v_measured,
)


print(
    f"Camera: {TARGET_IMAGE}"
)

print(
    f"Point: {TARGET_POINT}"
)


print(
    "\nValues read from "
    "calibration file:"
)

print(
    f"fx = {fx:.6f}"
)

print(
    f"fy = {fy:.6f}"
)

print(
    f"cx = {cx:.6f}"
)

print(
    f"cy = {cy:.6f}"
)

print(
    f"k1 = {k1:.10f}"
)

print(
    f"k2 = {k2:.10f}"
)

print(
    f"k3 = {k3:.10f}"
)

print(
    f"t1 = {t1:.10f}"
)

print(
    f"t2 = {t2:.10f}"
)


print(
    "\nMeasured pixel read "
    "from image-points file:"
)

print(
    f"u = {u_measured:.4f}, "
    f"v = {v_measured:.4f}"
)


print(
    "\nRecovered "
    "undistorted pixel:"
)

print(
    f"u = {u_undistorted:.4f}, "
    f"v = {v_undistorted:.4f}"
)


print(
    "\nRound-trip "
    "distorted pixel:"
)

print(
    f"u = {u_roundtrip:.4f}, "
    f"v = {v_roundtrip:.4f}"
)


print(
    f"\nRound-trip error: "
    f"{roundtrip_error:.10f} pixels"
)


object_point = object_points[
    object_points["GCP_label"]
    == TARGET_POINT
]

if len(object_point) != 1:
    raise ValueError(
        f"Expected exactly one object point "
        f"for {TARGET_POINT}, "
        f"but found {len(object_point)}."
    )

xyz_absolute = np.array(
    [
        object_point.iloc[0][
            "GCP_measured_X"
        ],
        object_point.iloc[0][
            "GCP_measured_Y"
        ],
        object_point.iloc[0][
            "GCP_measured_Z"
        ],
    ]
)


xyz_local = (
    xyz_absolute
    - offset
)


camera_position = camera[
    "camera_position"
]

rotation = camera[
    "rotation"
]

xyz_camera = rotation @ (
    xyz_local
    - camera_position
)

x_ideal = (
    xyz_camera[0]
    / xyz_camera[2]
)

y_ideal = (
    xyz_camera[1]
    / xyz_camera[2]
)

x_predicted_distorted, y_predicted_distorted = (
    distort_normalized(
        x_ideal,
        y_ideal,
    )
)

u_predicted = (
    fx * x_predicted_distorted
    + cx
)

v_predicted = (
    fy * y_predicted_distorted
    + cy
)

residual_u = (
    u_measured
    - u_predicted
)

residual_v = (
    v_measured
    - v_predicted
)

reprojection_error = np.hypot(
    residual_u,
    residual_v,
)

print(
    "\n--- Independent projection validation ---"
)

print(
    "\nAbsolute object coordinate:"
)

print(
    xyz_absolute
)

print(
    "\nEpoch offset:"
)

print(
    offset
)

print(
    "\nLocal object coordinate:"
)

print(
    xyz_local
)

print(
    "\nCamera position:"
)

print(
    camera_position
)

print(
    "\nCamera-space coordinate:"
)

print(
    xyz_camera
)

print(
    "\nPredicted distorted pixel:"
)

print(
    f"u = {u_predicted:.4f}, "
    f"v = {v_predicted:.4f}"
)

print(
    "\nMeasured distorted pixel:"
)

print(
    f"u = {u_measured:.4f}, "
    f"v = {v_measured:.4f}"
)

print(
    "\nPixel residuals:"
)

print(
    f"du = {residual_u:.4f} px"
)

print(
    f"dv = {residual_v:.4f} px"
)

print(
    f"\nReprojection error: "
    f"{reprojection_error:.4f} px"
)