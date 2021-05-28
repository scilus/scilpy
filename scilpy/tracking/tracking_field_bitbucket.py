
from __future__ import division

import logging

import dipy.core.geometry
import dipy.data
import dipy.reconst.shm
import numpy as np

from scilpy.reconst.utils import SphericalHarmonics
import scilpy.tracking.tools
import scilpy.utils.util


class TrackingDirection(list):

    """
    Tracking direction use as 3D cartesian direction (list(x,y,z))
    and has an index to work with discrete sphere.
    """

    def __init__(self, cartesian, index=None):
        super(TrackingDirection, self).__init__(cartesian)
        self.index = index


class AbstractField(object):

    """
    Interface to a tracking field.
    """

    def __init__(self, sf_threshold, sf_threshold_init, theta):
        self.dataset = None
        self.sf_threshold = sf_threshold
        self.sf_threshold_init = sf_threshold_init
        self.theta = theta
        self.cos_theta = np.cos(theta)

    def get_SF(self, pos):
        """
        return the spherical function at position pos.
        """
        pass

    def get_tracking_SF(self, pos, direction):
        """
        return the spherical function thresholded
        at position pos, for a direction.
        """
        pass

    def get_maxima(self, pos):
        """
        return the set of maxima at position pos from the thresholded SF.
        """
        pass

    def get_tracking_maxima(self, pos, direction):
        """
        return the set of maxima from the thresholded
        SF at position pos, for a direction.
        """
        pass

    def get_init_direction(self, pos):
        """
        return a tuple with an initial direction to follow from position pos,
        with the opposite direction.
        """
        pass

    def get_opposite_direction(self, direction):
        """
        return the opposite direction.
        """
        pass


class MaximaField(AbstractField):

    def __init__(self, maxima_dataset, sf_threshold, sf_threshold_init, theta):
        super(MaximaField, self).__init__(
            sf_threshold, sf_threshold_init, theta)
        self.dataset = maxima_dataset

    def get_SF(self, pos):
        scilpy.utils.util.not_implemented()

    def get_tracking_SF(self, pos, direction):
        peaks = self.dataset.getPositionValue(*pos).reshape(-1, 3)
        directions = []
        peaks_values = []
        for i in range(peaks.shape[0]):
            norm = np.linalg.norm(peaks[i])
            if norm > self.sf_threshold:
                d = peaks[i] / norm
                if np.dot(direction, d) > self.cos_theta:
                    directions.append(TrackingDirection(d))
                    peaks_values.append(norm)
                elif (np.dot(direction, self.get_opposite_direction(d))
                      > self.cos_theta):
                    directions.append(self.get_opposite_direction(d))
                    peaks_values.append(norm)
        return (np.array(peaks_values), directions)

    def get_maxima(self, pos, threshold=None):
        maxima = []
        peaks = self.dataset.getPositionValue(*pos).reshape(-1, 3)
        for i in range(peaks.shape[0]):
            if not np.isnan(peaks[i][0]) and not np.isnan(peaks[i][1]) \
                    and not np.isnan(peaks[i][2]):
                norm = np.linalg.norm(peaks[i])
                if norm > self.sf_threshold:
                    direction = TrackingDirection(list(peaks[i] / norm))
                    maxima.append(direction)
                    maxima.append(self.get_opposite_direction(direction))
        return maxima

    def get_tracking_maxima(self, pos, direction):
        valid_maxima = []
        maxima = self.get_maxima(pos)
        for m in maxima:
            if np.dot(direction, m) > self.cos_theta:
                valid_maxima.append(m)
        return valid_maxima

    def get_init_direction(self, pos):
        peaks = self.dataset.getPositionValue(*pos).reshape(-1, 3)
        dist = np.array([np.linalg.norm(p) for p in peaks])
        dist[dist < self.sf_threshold_init] = 0
        if np.sum(dist) > 0:
            ind = scilpy.tracking.tools.sample_distribution(dist)
            direction = TrackingDirection(peaks[ind] / dist[ind])
            return direction, self.get_opposite_direction(direction)
        return None, None

    def get_opposite_direction(self, direction):
        return TrackingDirection(np.array(direction) * -1)


class AbstractDiscreteField(AbstractField):

    def __init__(self, sf_threshold, sf_threshold_init, theta, dipy_sphere):
        super(AbstractDiscreteField, self).__init__(
            sf_threshold, sf_threshold_init, theta)

        self.vertices = dipy.data.get_sphere('symmetric724').vertices
        self.dirs = np.zeros(len(self.vertices), dtype=np.ndarray)
        for i in range(len(self.vertices)):
            self.dirs[i] = TrackingDirection(self.vertices[i], i)
        self.maxima_neighbours = self.get_direction_neighbours(np.pi / 16)
        self.tracking_neighbours = self.get_direction_neighbours(self.theta)

    def get_direction_neighbours(self, maxAngle):
        """
        return a matrix of neighbours for each direction on the sphere, within
        the maxAngle parameter.
        """
        xs = self.vertices[:, 0]
        ys = self.vertices[:, 1]
        zs = self.vertices[:, 2]
        scalarProds = np.outer(xs, xs) + np.outer(ys, ys) + np.outer(zs, zs)
        neighbours = scalarProds >= np.cos(maxAngle)
        return neighbours

    def get_opposite_direction(self, ind):
        """
        return the indice of the opposite direction on the sphere
        to the indice ind.
        """
        return (len(self.dirs) // 2 + ind) % len(self.dirs)

    def get_maxima(self, pos):
        SF = self.get_SF(pos)
        maxima = []
        for i in range(len(SF)):
            if np.max(SF[self.maxima_neighbours[i]]) == SF[i]:
                maxima.append(self.dirs[i])
        return maxima

    def get_tracking_maxima(self, pos, direction):
        SF = self.get_SF(pos)
        SF[SF < self.sf_threshold] = 0
        maxima = []
        for i in np.nonzero(self.tracking_neighbours[direction.index])[0]:
            if SF[i] > 0 and np.max(SF[self.maxima_neighbours[i]]) == SF[i]:
                maxima.append(self.dirs[i])
        return maxima

    def get_tracking_SF(self, pos, direction):
        SF = self.get_SF(pos)
        SF[SF < self.sf_threshold] = 0
        inds = np.nonzero(self.tracking_neighbours[direction.index])[0]
        return (SF[inds], self.dirs[inds])

    def get_init_direction(self, pos):
        SF = self.get_SF(pos)
        SF[SF < self.sf_threshold_init] = 0

        if np.sum(SF) > 0:
            ind = scilpy.tracking.tools.sample_distribution(SF)
            ind_opposite = self.get_opposite_direction(ind)
            return (self.dirs[ind], self.dirs[ind_opposite])
        return (None, None)


class SphericalHarmonicField(AbstractDiscreteField):

    def __init__(self, odf_dataset, basis, sf_threshold, sf_threshold_init,
                 theta, dipy_sphere='symmetric724'):
        super(SphericalHarmonicField, self).__init__(
            sf_threshold, sf_threshold_init, theta, dipy_sphere)
        self.dataset = odf_dataset
        if basis in ["dipy", "fibernavigator"]:
            logging.info('SH matrix computed using dipy basis.')
            self.basis = "descoteaux07"
        elif basis in ["mrtrix"]:
            logging.info('SH matrix computed using mrtrix basis.')
            self.basis = "tournier07"
        else:
            self.basis = basis

        sphere = dipy.data.get_sphere(dipy_sphere)
        self.SH = SphericalHarmonics(odf_dataset.data, self.basis, sphere)

    def get_SF(self, pos):
        SF = self.SH.get_SF(self.dataset.getPositionValue(*pos))

        SF_max = np.max(SF)
        if SF_max > 0:
            SF = SF / SF_max
        return SF


class SphericalFunctionField(AbstractDiscreteField):

    def __init__(self, sf_dataset, sf_threshold, sf_threshold_init, theta,
                 dipy_sphere='symmetric724'):
        super(SphericalFunctionField, self).__init__(
            sf_threshold, sf_threshold_init, theta, dipy_sphere)
        self.dataset = sf_dataset

    def get_SF(self, pos):
        SF = self.dataset.getPositionValue(*pos)
        SF_max = np.max(SF)
        if SF_max > 0:
            SF = SF / SF_max
        return SF
