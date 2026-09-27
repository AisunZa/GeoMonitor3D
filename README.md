# GeoMonitor3D

GeoMonitor3D is a Python workflow for multi-epoch deformation monitoring
using geodetic reference measurements and photogrammetric image observations.

The project reconstructs repeated monitoring points independently from
multi-view image measurements, compares 3D displacement vectors between
epochs, evaluates reprojection consistency and camera geometry, applies
robust triangulation, and investigates uncertainty using leave-one-out
and jackknife diagnostics.

## Dataset

The current implementation uses the HovoPhoto40 multi-epoch
photogrammetric deformation-monitoring dataset.

The present analysis compares the 2018 and 2024 epochs for the repeated
monitoring points:

`R1, R2, R3, R5, R6, R7`

The workflow reads the camera calibration, exterior orientation,
image-point observations, epoch offsets, and surveyed object coordinates
directly from the supplied dataset files.

## Processing workflow

The analysis includes:

1. Parsing calibrated camera parameters and image observations.
2. Correcting radial and tangential lens distortion.
3. Multi-ray 3D triangulation from image measurements.
4. Camera-ray geometry diagnostics.
5. Image-space reprojection residual analysis.
6. Leave-one-out observation screening.
7. Robust nonlinear triangulation using a Huber loss.
8. 2018–2024 photogrammetric displacement estimation.
9. Comparison with surveyed geodetic displacement.
10. Jackknife estimation of internal photogrammetric precision.
11. Full XYZ covariance propagation.
12. Sensitivity analysis to the assumed geodetic reference uncertainty.

## Main results

For the six repeated monitoring points, surveyed 2018–2024
3D displacement magnitudes are approximately 123–128 mm.

Using standard triangulation, the photogrammetric–geodetic
3D displacement-vector discrepancy has:

- mean: 3.928 mm
- RMSE: 4.180 mm
- maximum: 6.154 mm

Using robust Huber triangulation (`f_scale = 1.0 px`):

- mean: 3.596 mm
- RMSE: 3.846 mm
- maximum: 6.008 mm

The robust displacement-vector discrepancies range from:

- R3: 1.860 mm
- R5: 2.733 mm
- R2: 2.769 mm
- R6: 3.656 mm
- R7: 4.548 mm
- R1: 6.008 mm

Relative to the surveyed displacement magnitudes, these correspond
to approximately 1.5–4.8%.

## Robust estimation

Leave-one-out reprojection diagnostics identified particularly large
image-space inconsistencies for:

- R2 / 2024 / DSC04877
- R7 / 2024 / DSC04853

Rather than manually deleting observations, the final photogrammetric
solution uses nonlinear reprojection minimization with a Huber loss.

A sensitivity analysis using Huber scales of 0.5, 1.0, and 2.0 pixels
produced 3D displacement-vector RMSE values between approximately
3.78 and 3.91 mm, indicating that the result is not strongly dependent
on the exact robust-loss scale.

## Internal precision

Jackknife resampling was used to evaluate the sensitivity of each
reconstructed coordinate to its image observations.

The resulting internal 2018–2024 displacement precision scales range
from approximately:

`0.668 mm to 3.287 mm`

These values describe internal photogrammetric stability and should
not be interpreted as complete absolute accuracy estimates.

They do not fully account for uncertainty in camera calibration,
camera exterior orientation, reference-frame realization, or other
systematic effects.

## Uncertainty comparison

A full 3D covariance model was used to compare robust photogrammetric
and surveyed displacement vectors.

The dataset documentation reports approximately 4 mm accuracy for the
surveyed reference coordinates. Because this is not necessarily defined
as a one-standard-deviation uncertainty, the statistical comparison is
treated as a sensitivity analysis.

If the 4 mm value is provisionally interpreted as an independent
1-sigma coordinate uncertainty per epoch, all six
photogrammetric–geodetic displacement differences lie within the
95% full-covariance model.

Sensitivity analysis shows that all six points remain within that model
for assumed geodetic coordinate uncertainties of 2–4 mm per epoch.
At 1 mm per epoch, R1 and R7 fall outside the 95% region.

The largest full-covariance critical uncertainty threshold among the
six points is approximately 1.418 mm per coordinate per epoch.

This threshold is a sensitivity diagnostic and is not an estimate of
the actual survey uncertainty.

## Important interpretation

The supplied camera parameters may have been derived using surveyed
control information from the same photogrammetric project.

Therefore, comparison between reconstructed and surveyed coordinates
should currently be described as a photogrammetric–geodetic
consistency assessment rather than fully independent external validation,
unless the dataset methodology establishes that the evaluated monitoring
points were independent check points.

## Key outputs

Processed tables are written to:

`data/processed/`

Figures are written to:

`results/`

Important outputs include:

- `final_deformation_analysis_summary.csv`
- `robust_jackknife_precision.csv`
- `robust_jackknife_displacement_precision.csv`
- `loo_reprojection_diagnostics.csv`
- `standard_vs_robust_displacement_discrepancy.png`
- `robust_photogrammetric_vs_geodetic_vectors.png`
- `internal_precision_vs_external_discrepancy.png`

## Requirements

- Python
- NumPy
- pandas
- SciPy
- Matplotlib
- pytest

## Status

The core 2018–2024 deformation-analysis workflow is implemented.

Further work can extend the framework to the remaining HovoPhoto40
epochs and investigate fully independent validation and additional
camera-parameter uncertainty propagation.