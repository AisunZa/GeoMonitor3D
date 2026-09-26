# HovoPhoto40 Dataset Notes

## Selected initial epochs

- Baseline: 2018
- Comparison: 2024

## Measurement types

### Image point
2D pixel coordinate (x, y) of a target observed in an image.

### Object point
3D object-space coordinate (X, Y, Z) of a physical target.

### Ground Control Point
A target with independently known 3D coordinates used to establish
the photogrammetric reconstruction in physical space.

### Repeated reference points
R1, R2, R3, R5, R6, and R7 are available across all epochs and
can be used for multi-epoch change analysis.

### Internal camera orientation
Camera calibration parameters such as focal length, principal point,
and lens distortion.

### External camera orientation
Camera position (X, Y, Z) and rotation (omega, phi, kappa).

### Dense point cloud
A large set of reconstructed 3D surface points generated through
photogrammetric processing.

## Initial project strategy

1. Inspect 2018 geodetic/object coordinates.
2. Inspect 2024 geodetic/object coordinates.
3. Identify common reference points.
4. Calculate their measured 3D displacement.
5. Later derive displacement independently from photogrammetric data.
6. Compare photogrammetric displacement against the geodetic
   reference measurements.