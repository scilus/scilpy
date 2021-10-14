# -*- coding: utf-8 -*-
import dipy.data
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
import numpy as np

from scilpy.image.datasets import AccessibleVolume
from scilpy.tracking.tools import sample_distribution
from scilpy.tracking.utils import TrackingDirection


class SphericalHarmonicField(object):
    """
    Spherical harmonics tracking field.

    Parameters
    ----------
    odf_dataset: AccessibleVolume
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
    angle_maxima_detection: float
        Angle used for peak extraction to check if a direction is maximal in
        its neighbourhood.
    """

    def __init__(self, odf_dataset, basis, sf_threshold, sf_threshold_init,
                 theta, dipy_sphere='symmetric724',
                 angle_maxima_detection=np.pi / 16.):
        self.sf_threshold = sf_threshold
        self.sf_threshold_init = sf_threshold_init
        self.theta = theta

        self.vertices = dipy.data.get_sphere(dipy_sphere).vertices
        self.dirs = np.zeros(len(self.vertices), dtype=np.ndarray)
        for i in range(len(self.vertices)):
            self.dirs[i] = TrackingDirection(self.vertices[i], i)
        self.maxima_neighbours = self._get_direction_neighbours(
            angle_maxima_detection)
        self.tracking_neighbours = self._get_direction_neighbours(self.theta)
        self.dataset = odf_dataset
        self.basis = basis

        if 'symmetric' not in dipy_sphere:
            raise ValueError('Sphere must be symmetric. Call to '
                             'get_opposite_direction will fail.')

        sphere = dipy.data.get_sphere(dipy_sphere)
        sh_order = order_from_ncoef(self.dataset.data.shape[-1])
        self.B = sh_to_sf_matrix(sphere, sh_order, self.basis,
                                 smooth=0.006, return_inv=False)

    def _get_direction_neighbours(self, max_angle):
        """
        Get a matrix of neighbours for each direction on the sphere, within
        the maxAngle parameter.

        Parameters
        ----------
        max_angle: float
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
        scalar_prods = np.outer(xs, xs) + np.outer(ys, ys) + np.outer(zs, zs)
        neighbours = scalar_prods >= np.cos(max_angle)
        return neighbours

    def _get_sf(self, pos):
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
        sh = self.dataset.get_position_value(*pos)
        sf = np.dot(self.B.T, sh).reshape((-1, 1))

        sf_max = np.max(sf)
        if sf_max > 0:
            sf = sf / sf_max
        return sf

    def get_tracking_sf(self, pos, direction):
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
        sf = self._get_sf(pos)
        sf[sf < self.sf_threshold] = 0
        inds = np.nonzero(self.tracking_neighbours[direction.index])[0]
        return sf[inds], self.dirs[inds]

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
        sf = self._get_sf(pos)
        sf[sf < self.sf_threshold] = 0
        maxima = []
        for i in np.nonzero(self.tracking_neighbours[direction.index])[0]:
            if 0 < sf[i] == np.max(sf[self.maxima_neighbours[i]]):
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
        sf = self._get_sf(pos)
        sf[sf < self.sf_threshold_init] = 0

        if np.sum(sf) > 0:
            ind = sample_distribution(sf)
            ind_opposite = self.get_opposite_direction(ind)
            return self.dirs[ind], self.dirs[ind_opposite]
        return None, None

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
