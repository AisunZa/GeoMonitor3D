# Camera Model

The HovoPhoto40 calibrated camera parameter files provide:

- image width and height
- camera intrinsic matrix K
- radial distortion coefficients
- tangential distortion coefficients
- camera position
- camera rotation matrix

Both selected epochs use images of:

7952 x 5304 pixels

## 2018 intrinsic matrix

K =

[5362.6796, 0,         3957.3312
 0,         5362.6796, 2626.8057
 0,         0,         1]

## 2024 intrinsic matrix

K =

[5362.3087, 0,         3948.1732
 0,         5362.3087, 2651.9279
 0,         0,         1]

The focal lengths fx and fy are expressed in pixels.

The principal point (cx, cy) defines the calibrated optical center
in pixel coordinates.

Measured image points must be interpreted together with the camera
intrinsic parameters and distortion coefficients.

The general processing sequence is:

measured image pixel
→ normalized image coordinate
→ lens-distortion correction
→ camera ray
→ object-space ray
→ multi-view triangulation
→ 3D object coordinate