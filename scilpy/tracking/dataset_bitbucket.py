from __future__ import division

import numpy as np
from dipy.core.interpolation import trilinear_interpolate4d, \
                                    nearestneighbor_interpolate


class Dataset(object):

    """
    Class to access/interpolate data from nibabel object
    """

    def __init__(self, img, interpolation='trilinear'):
        self.interpolation = interpolation
        self.size = img.header['pixdim'][1:4]
        self.data = img.get_data(caching='unchanged').astype(np.float64)

        # Expand dimensionality to support uniform 4d interpolation
        if self.data.ndim == 3:
            self.data = np.expand_dims(self.data, axis=3)

        self.dim = self.data.shape[0:4]
        self.nbr_voxel = self.data.size

    def getVoxelValue(self, x, y, z):
        """
        get the voxel value at x, y, z in the dataset
        if the coordinates are out of bound, the nearest voxel value is taken.
        return: value
        """
        if not self.isVoxelInBound(x, y, z):
            x = max(0, min(self.dim[0] - 1, x))
            y = max(0, min(self.dim[1] - 1, y))
            z = max(0, min(self.dim[2] - 1, z))

        return self.data[x][y][z]

    def isVoxelInBound(self, x, y, z):
        """
        return: true if voxel is in dataset range
        return false otherwise
        """
        return (x < self.dim[0] and y < self.dim[1] and z < self.dim[2] and
                x >= 0 and y >= 0 and z >= 0)

    def getVoxelAtPosition(self, x, y, z):
        """
        return: integer value of position/dimention
        """
        return [(x + self.size[0] / 2) // self.size[0],
                (y + self.size[1] / 2) // self.size[1],
                (z + self.size[2] / 2) // self.size[2]]

    def getVoxelCoordinate(self, x, y, z):
        """
        return: value of position/dimention
        """
        return [x / self.size[0], y / self.size[1], z / self.size[2]]

    def getVoxelValueAtPosition(self, x, y, z):
        """
        get the voxel value at position x, y, z in the dataset
        return: value
        """
        return self.getVoxelValue(*self.getVoxelAtPosition(x, y, z))

    def getPositionValue(self, x, y, z):
        """
        get the voxel value at voxel coordinate x, y, z in the dataset
        if the coordinates are out of bound, the nearest voxel value is taken.
        return value
        """
        if not self.isPositionInBound(x, y, z):
            eps = float(1e-8)  # Epsilon to exclude upper borders
            x = max(-self.size[0] / 2,
                    min(self.size[0] * (self.dim[0] - 0.5 - eps), x))
            y = max(-self.size[1] / 2,
                    min(self.size[1] * (self.dim[1] - 0.5 - eps), y))
            z = max(-self.size[2] / 2,
                    min(self.size[2] * (self.dim[2] - 0.5 - eps), z))
        coord = np.array(self.getVoxelCoordinate(x, y, z), dtype=np.float64)

        if self.interpolation == 'nearest':
            result = nearestneighbor_interpolate(self.data, coord)
        elif self.interpolation == 'trilinear':
            result = trilinear_interpolate4d(self.data, coord)
        else:
            raise Exception("Invalid interpolation method.")

        # Squeezing returns only value instead of array of length 1 if 3D data
        return np.squeeze(result)

    def isPositionInBound(self, x, y, z):
        """
        return: true if position is in dataset range
        return false otherwise
        """
        return self.isVoxelInBound(*self.getVoxelAtPosition(x, y, z))
