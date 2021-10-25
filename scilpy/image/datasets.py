# -*- coding: utf-8 -*-
import numpy as np

from dipy.core.interpolation import trilinear_interpolate4d, \
                                    nearestneighbor_interpolate


class DataVolume(object):
    """
    Class to access/interpolate data from nibabel object
    """

    def __init__(self, img, interpolation=None, must_be_3d=False):
        """
        Parameters
        ----------
        img: nibabel image
            The nibabel image from which to get the data
        interpolation: str or None
            The interpolation choice amongst "trilinear" or "nearest". If
            None, functions getting a coordinate in mm instead of voxel
            coordinates are not available.
        must_be_3d: bool
            If True, dataset can't be 4D.
        """
        self.interpolation = interpolation
        if self.interpolation:
            if not (self.interpolation == 'trilinear' or
                    self.interpolation == 'nearest'):
                raise Exception("Interpolation must be 'trilinear' or "
                                "'nearest'")

        self.pixdim = img.header.get_zooms()[:3]
        self.data = img.get_fdata(caching='unchanged', dtype=np.float64)

        if must_be_3d and self.data.ndim != 3:
            raise Exception("Data should have been 3D but data dimension is:"
                            "{}".format(self.data.ndim))

        # Expand dimensionality to support uniform 4d interpolation
        if self.data.ndim == 3:
            self.data = np.expand_dims(self.data, axis=3)

        self.dim = self.data.shape[0:4]
        self.nbr_voxel = self.data.size

    def get_voxel_value(self, i, j, k):
        """
        Get the voxel value at x, y, z in the dataset
        if the coordinates are out of bound, the nearest voxel
        value is taken.

        Parameters
        ----------
        i, j, k: ints
            Voxel indice along each axis.

        Return
        ------
        value: ndarray (self.dim[-1],)
            The value evaluated at voxel x, y, z.
        """
        if not self.is_voxel_in_bound(i, j, k):
            i = max(0, min(self.dim[0] - 1, i))
            j = max(0, min(self.dim[1] - 1, j))
            k = max(0, min(self.dim[2] - 1, k))

        return self.data[i][j][k]

    def is_voxel_in_bound(self, i, j, k):
        """
        Test if voxel is in dataset range.

        Parameters
        ----------
        i, j, k: ints
            Voxel indice along each axis.

        Return
        ------
        out: bool
            True if voxel is in dataset range, False otherwise.
        """
        return (0 <= i < self.dim[0] and 0 <= j < self.dim[1] and
                0 <= k < self.dim[2])

    def get_voxel_at_position(self, x, y, z):
        """
        Get the 3D indice of the closest voxel at position x, y, z expressed
        in mm.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.

        Return
        ------
        out: list
            3D indice of voxel at position x, y, z.
        """
        return [(x + self.pixdim[0] / 2) // self.pixdim[0],
                (y + self.pixdim[1] / 2) // self.pixdim[1],
                (z + self.pixdim[2] / 2) // self.pixdim[2]]

    def get_voxel_coordinate(self, x, y, z):
        """
        Get voxel space coordinates at position x, y, z (mm).

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.

        Return
        ------
        out: list
            Voxel space coordinates for position x, y, z.
        """
        return [x / self.pixdim[0], y / self.pixdim[1], z / self.pixdim[2]]

    def get_voxel_value_at_position(self, x, y, z):
        """
        Get value of the voxel closest to position x, y, z (mm) in the dataset.
        No interpolation is done.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.

        Return
        ------
        value: ndarray (self.dim[-1],)
            The value evaluated at position x, y, z.
        """
        return self.get_voxel_value(*self.get_voxel_at_position(x, y, z))

    def get_position_value(self, x, y, z):
        """
        Get the voxel value at voxel position x, y, z (mm) in the dataset.
        If the coordinates are out of bound, the nearest voxel value is taken.
        Value is interpolated based on the value of self.interpolation.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.

        Return
        ------
        value: ndarray (self.dims[-1],) or float
            Interpolated value at position x, y, z (mm). If the last dimension
            is of length 1, return a scalar value.
        """
        if self.interpolation is not None:
            if not self.is_position_in_bound(x, y, z):
                eps = float(1e-8)  # Epsilon to exclude upper borders
                x = max(-self.pixdim[0] / 2,
                        min(self.pixdim[0] * (self.dim[0] - 0.5 - eps), x))
                y = max(-self.pixdim[1] / 2,
                        min(self.pixdim[1] * (self.dim[1] - 0.5 - eps), y))
                z = max(-self.pixdim[2] / 2,
                        min(self.pixdim[2] * (self.dim[2] - 0.5 - eps), z))
            coord = np.array(self.get_voxel_coordinate(x, y, z),
                             dtype=np.float64)

            if self.interpolation == 'nearest':
                result = nearestneighbor_interpolate(self.data, coord)
            else:
                # Trilinear
                result = trilinear_interpolate4d(self.data, coord)

            # Squeezing returns only value instead of array of length 1 if 3D
            # data
            return np.squeeze(result)

        else:
            raise Exception("No interpolation method was given, cannot run "
                            "this method..")

    def is_position_in_bound(self, x, y, z):
        """
        Test if the position x, y, z mm is in the dataset range.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.

        Return
        ------
        value: bool
            True if position is in dataset range and false otherwise.
        """
        return self.is_voxel_in_bound(*self.get_voxel_at_position(x, y, z))
