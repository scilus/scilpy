# -*- coding: utf-8 -*-
import logging

import dipy.data
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
import numpy as np

from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.tracking.tools import sample_distribution
from scilpy.tracking.utils import TrackingDirection


class SphericalHarmonicField(object):
    """
    Spherical harmonics tracking field.

    Parameters
    ----------
    odf_dataset: scilpy Dataset
        Trackable Dataset object.
    basis: string
        SH basis name. One of 'tournier07' or 'descoteaux07'
    sf_threshold: float
        Threshold on spherical function (SF).
    sf_threshold_init: float
        Threshold on spherical function when initializing a new streamline.
    theta: float
        Maximum angle (radians) between two steps.
    dipy_sphere: string
        Name of the DIPY sphere object to use for evaluating SH.
    """

    def __init__(self, odf_dataset, basis, sf_threshold, sf_threshold_init,
                 theta, dipy_sphere='symmetric724'):
        self.sf_threshold = sf_threshold
        self.sf_threshold_init = sf_threshold_init
        self.theta = theta

        self.vertices = dipy.data.get_sphere(dipy_sphere).vertices
        self.dirs = np.zeros(len(self.vertices), dtype=np.ndarray)
        for i in range(len(self.vertices)):
            self.dirs[i] = TrackingDirection(self.vertices[i], i)
        self.maxima_neighbours = self.get_direction_neighbours(np.pi / 16.)
        self.tracking_neighbours = self.get_direction_neighbours(self.theta)
        self.dataset = odf_dataset
        self.basis = basis

        if 'symmetric' not in dipy_sphere:
            raise ValueError('Sphere must be symmetric. Call to '
                             'get_opposite_direction will fail.')

        self.sphere = dipy.data.get_sphere(dipy_sphere)
        self.sh_order, self.full_basis =\
            get_sh_order_and_fullness(self.dataset.data.shape[-1])
        self.B = sh_to_sf_matrix(self.sphere, self.sh_order, self.basis,
                                 full_basis=self.full_basis, smooth=0.006,
                                 return_inv=False)
        self.output_sf_image_suffix = 0

    def get_direction_neighbours(self, maxAngle):
        """
        Get a matrix of neighbours for each direction on the sphere, within
        the maxAngle parameter.

        Parameters
        ----------
        maxAngle: float
            Maximum angle in radians defining the neighbourhood
            of each direction.

        Return
        ------
        neighbours: ndarray
            Neighbour directions for each direction on the sphere.
        """
        xs = self.vertices[:, 0]
        ys = self.vertices[:, 1]
        zs = self.vertices[:, 2]
        scalarProds = np.outer(xs, xs) + np.outer(ys, ys) + np.outer(zs, zs)
        neighbours = scalarProds >= np.cos(maxAngle)
        return neighbours

    def get_SF(self, pos):
        """
        Get the spherical function at position pos.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in mm in the trackable dataset.

        Return
        ------
        sf: ndarray (len(self.vertices),)
            Spherical function evaluated at pos, normalized by
            its maximum amplitude.
        """
        sh = self.dataset.getPositionValue(*pos)
        sf = np.dot(self.B.T, sh).reshape((-1, 1))

        sf_max = np.max(sf)
        if sf_max > 0:
            sf = sf / sf_max
        return sf

    def get_tracking_SF(self, pos, direction):
        """
        Get the spherical functions thresholded
        at position pos, for a given direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset, expressed in mm.
        direction: TrackingDirection
            A given direction.

        Return
        ------
        value: tuple
            The neighbours SF evaluated at pos in given direction and
            corresponding tracking directions.
        """
        SF = self.get_SF(pos)
        SF[SF < self.sf_threshold] = 0
        inds = np.nonzero(self.tracking_neighbours[direction.index])[0]
        return (SF[inds], self.dirs[inds])

    def get_maxima(self, pos):
        """
        Get the set of maxima at position pos from the thresholded SF.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset, expressed in mm.

        Return
        ------
        maxima: list
            Set of maxima directions at position pos.
        """
        SF = self.get_SF(pos)
        maxima = []
        for i in range(len(SF)):
            if np.max(SF[self.maxima_neighbours[i]]) == SF[i]:
                maxima.append(self.dirs[i])
        return maxima

    def get_tracking_maxima(self, pos, direction):
        """
        Get the set of maxima directions from the thresholded
        SF at position pos, for a direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset, expressed in mm.
        direction: TrackingDirection
            A given direction.

        Return
        ------
        maxima: list
            List of directions of maxima around the input direction at pos.
        """
        SF = self.get_SF(pos)
        SF[SF < self.sf_threshold] = 0
        maxima = []
        for i in np.nonzero(self.tracking_neighbours[direction.index])[0]:
            if SF[i] > 0 and np.max(SF[self.maxima_neighbours[i]]) == SF[i]:
                maxima.append(self.dirs[i])
        return maxima

    def get_init_direction(self, pos):
        """
        Get a tuple with an initial direction to follow from position pos,
        with the opposite direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset, expressed in mm.

        Return
        ------
        value: tuple
            Initial direction to follow from pos and its opposite direction.
        """
        SF = self.get_SF(pos)
        SF[SF < self.sf_threshold_init] = 0

        if np.sum(SF) > 0:
            ind = sample_distribution(SF)
            ind_opposite = self.get_opposite_direction(ind)
            return (self.dirs[ind], self.dirs[ind_opposite])
        return (None, None)

    def get_opposite_direction(self, ind):
        """
        Get the indice of the opposite direction on the sphere
        to the indice ind.

        Parameters
        ----------
        ind: int
            Indice of sphere direction

        Return
        ------
        value: int
            Indice of opposite sphere direction.
        """
        return (len(self.dirs) // 2 + ind) % len(self.dirs)
