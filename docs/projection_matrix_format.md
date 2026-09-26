# Projection Matrix and Coordinate Offset

Each row of the HovoPhoto40 pmatrix file contains:

- image filename
- 12 projection-matrix coefficients

The 12 coefficients form a 3x4 camera projection matrix:

P = [p11 p12 p13 p14
     p21 p22 p23 p24
     p31 p32 p33 p34]

The matrix maps a 3D object point in the local photogrammetric
coordinate system to homogeneous image coordinates.

The epoch-specific offsets are:

2018:
X0 = 455.000 m
Y0 = 434.000 m
Z0 = 38.000 m

2024:
X0 = 442.000 m
Y0 = 424.000 m
Z0 = 36.000 m

Before projection, object coordinates are expressed relative to
the corresponding epoch offset:

X_local = X - X0
Y_local = Y - Y0
Z_local = Z - Z0

The projection produces homogeneous coordinates (u, v, w):

[u, v, w]^T = P [X_local, Y_local, Z_local, 1]^T

Image coordinates are then:

x = u / w
y = v / w

The offsets are computational coordinate origins and must not be
interpreted as physical deformation between epochs.