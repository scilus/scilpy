# -*- coding: utf-8 -*-
import logging

import dipy.data
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
import numpy as np

import scilpy.tracking.tools


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
        self.basis = basis

        sphere = dipy.data.get_sphere(dipy_sphere)
        sh_order = order_from_ncoef(self.dataset.data.shape[-1])
        self.B, self.invB = sh_to_sf_matrix(sphere, sh_order, self.basis,
                                            smooth=0.006)

    def get_SF(self, pos):
        sh = self.dataset.getPositionValue(*pos)
        sf = np.dot(self.B.T, sh).reshape((-1, 1))

        sf_max = np.max(sf)
        if sf_max > 0:
            sf = sf / sf_max
        return sf
