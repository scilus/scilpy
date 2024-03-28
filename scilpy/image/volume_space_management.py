# -*- coding: utf-8 -*-
import numpy as np

from dipy.core.interpolation import trilinear_interpolate4d, \
    nearestneighbor_interpolate
from dipy.io.stateful_tractogram import Origin, Space


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

    def get_value_at_idx(self, i, j, k):
        """
        Get the voxel value at index i, j, k in the dataset.
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
        i, j, k = self._clip_idx_to_bound(i, j, k)
        return self.data[i][j][k]

    def get_value_at_coordinate(self, x, y, z, space, origin):
        """
        Get the voxel value at coordinates x, y, z, in the dataset.
        Coordinates must be in the given space and origin.

        If the coordinates are out of bound, the nearest voxel value is taken.

        Parameters
        ----------
        x, y, z: floats
            Voxel coordinates along each axis.
        space: dipy Space
            'vox' or 'voxmm'.
        origin: dipy Origin
            'corner' or 'center'.

        Return
        ------
        value: ndarray (self.dim[-1],)
            The value evaluated at voxel x, y, z.
        """
        if space == Space.VOX:
            return self._vox_to_value(x, y, z, origin)
        elif space == Space.VOXMM:
            return self._voxmm_to_value(x, y, z, origin)
        else:
            raise NotImplementedError("We have not prepared the DataVolume to "
                                      "work in RASMM space yet.")

    def is_idx_in_bound(self, i, j, k):
        """
        Test if voxel is in dataset range.

        Parameters
        ----------
        i, j, k: ints
            Voxel indice along each axis (as ints).

        Return
        ------
        out: bool
            True if voxel is in dataset range, False otherwise.
        """
        return (0 <= i < (self.dim[0]) and
                0 <= j < (self.dim[1]) and
                0 <= k < (self.dim[2]))

    def is_coordinate_in_bound(self, x, y, z, space, origin):
        """
        Test if voxel is in dataset range.

        Parameters
        ----------
        x, y, z: floats
            Voxel coordinates along each axis.
        space: dipy Space
            'vox' or 'voxmm'.
        origin: dipy Origin
            'corner' or 'center'.

        Return
        ------
        out: bool
            True if voxel is in dataset range, False otherwise.
        """
        if space == Space.VOX:
            return self._is_vox_in_bound(x, y, z, origin)
        elif space == Space.VOXMM:
            return self._is_voxmm_in_bound(x, y, z, origin)
        else:
            raise NotImplementedError("We have not prepared the DataVolume to "
                                      "work in RASMM space yet.")

    def _clip_idx_to_bound(self, i, j, k):
        """
        Returns i, j, k if the index is valid inside the bounding box. Else,
        finds the closest valid index on the border.
        """
        if not self.is_idx_in_bound(i, j, k):
            i = max(0, min(self.dim[0] - 1, i))
            j = max(0, min(self.dim[1] - 1, j))
            k = max(0, min(self.dim[2] - 1, k))
        return i, j, k

    def _clip_vox_to_bound(self, x, y, z, origin):
        """
        Returns x, y, z if the voxel coordinate is valid inside the bounding
        box. Else, finds the closest valid value on the border.

        Parameters
        ----------
        x, y, z: floats
            Current coordinates (vox).
        origin: Dipy space
            'center' or 'corner'.

        Return
        ------
        x, y, z: floats
            Closest valid coordinates.
        """
        if not self._is_vox_in_bound(x, y, z, origin):
            eps = float(1e-8)  # Epsilon to exclude upper borders
            if origin == Origin('corner'):
                # Min = 0
                # Max = 0.9999 in first voxel.
                x = max(0, min(self.dim[0] - eps, x))
                y = max(0, min(self.dim[1] - eps, y))
                z = max(0, min(self.dim[2] - eps, z))
            elif origin == Origin('center'):
                # Min = -0.5
                # Max = 0.499999 in first voxel.
                x = max(-0.5, min(self.dim[0] - 0.5 - eps, x))
                y = max(-0.5, min(self.dim[1] - 0.5 - eps, y))
                z = max(-0.5, min(self.dim[2] - 0.5 - eps, z))
            else:
                raise ValueError("Origin should be 'center' or 'corner'.")
        return x, y, z

    @staticmethod
    def vox_to_idx(x, y, z, origin):
        """
        Get the 3D indice of the closest voxel at position x, y, z expressed
        in mm.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate in voxel space along x, y, z axis.
        origin: Dipy Space
            'center' or 'corner'.

        Return
        ------
        out: list
            3D indice of voxel at position x, y, z.
        """
        if origin == Origin('corner'):
            return np.floor((x, y, z))
        elif origin == Origin('center'):
            return np.floor((x + 0.5, y + 0.5, z + 0.5))
        else:
            raise ValueError("Origin must be 'center' or 'corner'.")

    def _vox_to_value(self, x, y, z, origin):
        """
        Get the voxel value at voxel position x, y, z (vox) in the dataset.
        If the coordinates are out of bound, the nearest voxel value is taken.
        Value is interpolated based on the value of self.interpolation.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (vox) along x, y, z axis.
        origin: dipy Space
            'center' or 'corner'.

        Return
        ------
        value: ndarray (self.dims[-1],) or float
            Interpolated value at position x, y, z (mm). If the last dimension
            is of length 1, return a scalar value.
        """
        if self.interpolation is not None:
            # Checking if out of bound.
            x, y, z = self._clip_vox_to_bound(x, y, z, origin)

            # Interpolation: Using dipy's pyx methods. The doc can be found in
            # the file dipy.core.interpolation.pxd. Dipy works with origin
            # center.
            # Note. Data is expected to be double (float64), can't use float32.
            coord = np.array((x, y, z), dtype=np.float64)
            if origin == Origin('corner'):
                coord -= 0.5

            if self.interpolation == 'nearest':
                # They use round(point), not floor. This is the equivalent of
                # origin = 'center'.
                result = nearestneighbor_interpolate(self.data, coord)
            else:
                # Trilinear
                # They do not say it explicitly but they verify if
                # point[i] < -.5 or point[i] >= (data.shape[i] - .5),
                # meaning that they work with origin='center'.
                result = trilinear_interpolate4d(self.data, coord)

            # Squeezing returns only value instead of array of length 1 if 3D
            # data
            return np.squeeze(result)
        else:
            raise Exception("No interpolation method was given, cannot run "
                            "this method..")

    def _is_vox_in_bound(self, x, y, z, origin):
        """
        Test if voxel is in dataset range.

        Parameters
        ----------
        x, y, z: floats
            Voxel coordinates along each axis in voxel space.
        origin: dipy Space
            Origin ('center' or 'corner').

        Return
        ------
        out: bool
            True if voxel is in dataset range, False otherwise.
        """
        return self.is_idx_in_bound(*self.vox_to_idx(x, y, z, origin))

    def voxmm_to_idx(self, x, y, z, origin):
        """
        Get the 3D indice of the closest voxel at position x, y, z expressed
        in mm.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.
        origin: dipy Space
            'center' or 'corner'.

        Return
        ------
        x, y, z: ints
            3D indice of voxel at position x, y, z.
        """
        return self.vox_to_idx(*self.voxmm_to_vox(x, y, z), origin)

    def voxmm_to_vox(self, x, y, z):
        """
        Get voxel space coordinates at position x, y, z (mm).

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.

        Return
        ------
        x, y, z: floats
            Voxel space coordinates for position x, y, z.
        """
        # Does not depend on origin!
        # In each dimension:
        #   In corner: 0 to voxres will become 0 to 1.
        #   In center: -0.5*voxres to 0.5*voxres will become -0.5 to 0.5.
        return [x / self.voxres[0],
                y / self.voxres[1],
                z / self.voxres[2]]

    def _voxmm_to_value(self, x, y, z, origin):
        """
        Get the voxel value at voxel position x, y, z (mm) in the dataset.
        If the coordinates are out of bound, the nearest voxel value is taken.
        Value is interpolated based on the value of self.interpolation.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.
        origin: dipy Space
            'center' or 'corner'.

        Return
        ------
        value: ndarray (self.dims[-1],) or float
            Interpolated value at position x, y, z (mm). If the last dimension
            is of length 1, return a scalar value.
        """
        return self._vox_to_value(*self.voxmm_to_vox(x, y, z), origin)

    def _is_voxmm_in_bound(self, x, y, z, origin):
        """
        Test if the position x, y, z mm is in the dataset range.

        Parameters
        ----------
        x, y, z: floats
            Position coordinate (mm) along x, y, z axis.
        origin: dipy Space
            'Center': Voxel 0,0,0 goes from [-resx/2, -resy/2, -resz/2] to
                [resx/2, resy/2, resz/2].
            'Corner': Voxel 0,0,0 goes from [0,0,0] to [resx, resy, resz].

        Return
        ------
        value: bool
            True if position is in dataset range and false otherwise.
        """
        return self.is_idx_in_bound(*self.voxmm_to_idx(x, y, z, origin))
