# Image Point File Format

The HovoPhoto40 image-point files contain no header row.

Each line has four fields:

1. image_name
2. point_label
3. image_x
4. image_y

Example:

DSC02875,R1,6893.7666,972.2086

This means that physical target R1 was measured in image DSC02875
at image coordinates:

x = 6893.7666
y = 972.2086

The same physical target may appear in multiple images with different
image coordinates.

Object-space coordinates (X, Y, Z) describe the physical 3D target
location, while image coordinates (x, y) describe where that target
appears in an individual photograph.

Multiple observations of the same target from different camera
positions can later be used for photogrammetric triangulation.