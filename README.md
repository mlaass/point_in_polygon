# PCL - Point Cloud Library
point cloud library with the following functions

- Boost Graph Library for Environments (e.g. Rank k)
- HDF5 (plain, could be streamlined with HighFive)
- Boost Geometry
- Boost ranges
- Eigen (dense matrix and eigenvalues) libeigen3-dev


Then we create `flat_point` data structures and bind them as geometry to a coordinate system - so we can run boost geometry on memory areas.


`value_maker` is created on bulk loading to make a point a box - then we can bulk load r-tree without copy.

Zonal-Key is then intended for 2D extracts (e.g. Twitter) is only of limited use here, but can be generalized to 3D.

You will then find the structure tensors in the namespace mpcl

And the core of the library is the pointcloud class.

extractKNN can make your kNN environments.

Line 669 ff then creates the point cloud features for this environment.

You then have features as a property map on your points, whereby we do not calculate the tensors on the points, but on the centroid of the k nearest neighbors. You would have to add range (all points in the radius r).

