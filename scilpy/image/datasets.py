# -*- coding: utf-8 -*-
import numpy as np

from dipy.core.interpolation import trilinear_interpolate4d, \
    nearestneighbor_interpolate


class DataVolume(object):
    """
    Class to access/interpolate data from nibabel object
    """

    def __init__(self, data, voxres, interpolation=None, must_be_3d=False):
        """
        Parameters
        ----------
        data: np.array
            The data, ex, loaded from nibabel img.get_fdata().
        voxres: np.array(3,)
            The pixel resolution, ex, using img.header.get_zooms()[:3].
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

        self.data = data
        self.voxres = voxres

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
        Get the voxel value at x, y, z in the dataset.
        If the coordinates are out of bound, the nearest voxel value is taken.

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
        i, j, k: ints or floats
            Voxel indice along each axis (as ints) or voxel coordinates in
            voxel world (as floats).

        Return
        ------
        out: bool
            True if voxel is in dataset range, False otherwise.
        """
        return (0 <= i < (self.dim[0]) and
                0 <= j < (self.dim[1]) and
                0 <= k < (self.dim[2]))

    def voxmm_to_idx(self, x, y, z, origin):
        """
        Get the 3D indice of the closest voxel at position x, y, z expressed
        in mm.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.
        origin: str
            'center': Voxel 0,0,0 goes from [-resx/2, -resy/2, -resz/2] to
                [resx/2, resy/2, resz/2].
            'corner': Voxel 0,0,0 goes from [0,0,0] to [resx, resy, resz].

        Return
        ------
        out: list
            3D indice of voxel at position x, y, z.
        """
        return np.floor(self.voxmm_to_vox(x, y, z, origin))

    def voxmm_to_vox(self, x, y, z, origin):
        """
        Get voxel space coordinates at position x, y, z (mm).

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.
        origin: str
            'center': Voxel 0,0,0 goes from [-resx/2, -resy/2, -resz/2] to
                [resx/2, resy/2, resz/2].
            'corner': Voxel 0,0,0 goes from [0,0,0] to [resx, resy, resz].

        Return
        ------
        out: list
            Voxel space coordinates for position x, y, z.
        """
        if origin == 'center':
            half_res = self.voxres / 2.
            return [(x + half_res[0]) / self.voxres[0],
                    (y + half_res[1]) / self.voxres[1],
                    (z + half_res[2]) / self.voxres[2]]
        elif origin == 'corner':
            return [x / self.voxres[0],
                    y / self.voxres[1],
                    z / self.voxres[2]]
        else:
            raise ValueError("Origin should be 'center' or 'corner'.")

    def voxmm_to_value(self, x, y, z, origin):
        """
        Get the voxel value at voxel position x, y, z (mm) in the dataset.
        If the coordinates are out of bound, the nearest voxel value is taken.
        Value is interpolated based on the value of self.interpolation.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.
        origin: str
            'center': Voxel 0,0,0 goes from [-resx/2, -resy/2, -resz/2] to
                [resx/2, resy/2, resz/2].
            'corner': Voxel 0,0,0 goes from [0,0,0] to [resx, resy, resz].

        Return
        ------
        value: ndarray (self.dims[-1],) or float
            Interpolated value at position x, y, z (mm). If the last dimension
            is of length 1, return a scalar value.
        """
        if self.interpolation is not None:
            if not self.is_voxmm_in_bound(x, y, z, origin):
                eps = float(1e-8)  # Epsilon to exclude upper borders
                if origin == 'corner':
                    x = max(0,
                            min(self.voxres[0] * (self.dim[0] - eps), x))
                    y = max(0,
                            min(self.voxres[1] * (self.dim[1] - eps), y))
                    z = max(0,
                            min(self.voxres[2] * (self.dim[2] - eps), z))
                elif origin == 'center':
                    x = max(-self.voxres[0] / 2,
                            min(self.voxres[0] * (self.dim[0] - 0.5 - eps), x))
                    y = max(-self.voxres[1] / 2,
                            min(self.voxres[1] * (self.dim[1] - 0.5 - eps), y))
                    z = max(-self.voxres[2] / 2,
                            min(self.voxres[2] * (self.dim[2] - 0.5 - eps), z))
                else:
                    raise ValueError("Origin should be 'center' or 'corner'.")

            coord = np.array(self.voxmm_to_vox(x, y, z, origin),
                             dtype=np.float64)

            # Interpolation: Using dipy's pyx methods. The doc can be found in
            # the file dipy.core.interpolation.pxd. Dipy works with origin
            # center.
            if origin == 'corner':
                coord -= 0.5
            if self.interpolation == 'nearest':
                # They use round(point), not floor. This is the equivalent of
                # origin = 'center'.
                result = nearestneighbor_interpolate(self.data, coord)
            else:
                # Trilinear
                # They do not say it explicitely but they verify if
                # point[i] < -.5 or point[i] >= (data.shape[i] - .5),
                # meaning that they work with origin='center'.
                result = trilinear_interpolate4d(self.data, coord)

            # Squeezing returns only value instead of array of length 1 if 3D
            # data
            return np.squeeze(result)
        else:
            raise Exception("No interpolation method was given, cannot run "
                            "this method..")

    def is_voxmm_in_bound(self, x, y, z, origin):
        """
        Test if the position x, y, z mm is in the dataset range.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.
        origin: str
            'Center': Voxel 0,0,0 goes from [-resx/2, -resy/2, -resz/2] to
                [resx/2, resy/2, resz/2].
            'Corner': Voxel 0,0,0 goes from [0,0,0] to [resx, resy, resz].

        Return
        ------
        value: bool
            True if position is in dataset range and false otherwise.
        """
        return self.is_voxel_in_bound(*self.voxmm_to_vox(x, y, z, origin))
