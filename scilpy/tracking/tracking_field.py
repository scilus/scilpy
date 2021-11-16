# -*- coding: utf-8 -*-
import logging

import dipy.data
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
import numpy as np

from scilpy.tracking.tools import sample_distribution
from scilpy.tracking.utils import TrackingDirection


class AbstractTrackingField(object):
    """
    Abstract tracking field. The class is used to get directions from the
    dataset (ex, peaks, ODF, fODF, machine learning models), and returns only
    the subset of possible directions that are inside a cone of angle theta.

    Parameters
    ----------
    dataset: Any
        Dataset object.
    theta: float
        Maximum angle (radians) between two steps.
    dipy_sphere: string, optional
        If necessary, name of the DIPY sphere object to use to evaluate
        directions.
    """

    def __init__(self, dataset, theta, dipy_sphere=None):
        self.theta = theta
        self.dataset = dataset

        if dipy_sphere:
            if 'symmetric' not in dipy_sphere:
                raise ValueError('Sphere must be symmetric. Call to '
                                 'get_opposite_direction will fail.')
            self.sphere = dipy.data.get_sphere(dipy_sphere)
            self.dirs = np.zeros(len(self.sphere.vertices), dtype=np.ndarray)
            for i in range(len(self.sphere.vertices)):
                self.dirs[i] = TrackingDirection(self.sphere.vertices[i], i)
            self.tracking_neighbours = self._get_sphere_neighbours(self.theta)
        else:
            self.sphere = None
            self.dirs = []
            self.tracking_neighbours = None

    def _get_sphere_neighbours(self, max_angle):
        """
        Get a matrix of neighbours for each direction on the sphere, within
        the min_separation_angle.

        min_separation_angle: float
            Maximum angle in radians defining the neighbourhood
            of each direction.

        Return
        ------
        neighbours: ndarray
            Neighbour directions for each direction on the sphere.
        """
        if self.sphere is not None:
            xs = self.sphere.vertices[:, 0]
            ys = self.sphere.vertices[:, 1]
            zs = self.sphere.vertices[:, 2]
            scalar_prods = (np.outer(xs, xs) + np.outer(ys, ys) +
                            np.outer(zs, zs))
            neighbours = scalar_prods >= np.cos(max_angle)
            return neighbours
        else:
            logging.warning("Sphere is not set. Can't get sphere neighbors.")
            return None

    def get_init_direction(self, pos):
        """
        Get a tuple with an initial direction to follow from position pos,
        with the opposite direction.

        Must be instantiated by each child class.

        Parameters
        ----------
        pos: Any
            Current position in the dataset.
        """
        raise NotImplementedError

    def get_possible_next_directions(self, pos, previous_direction, *args):
        """
        Get the set of next possible directions.
        Depends on the type of tracking field and probably on some algorithm
        parameter choices.

        The way to sample from these possible directions is the propagator's
        job.

        Must be instantiated by each child class.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset, expressed in mm.
        previous_direction: TrackingDirection
            Incoming direction. Outcoming direction won't be further than an
            angle theta.
        args: Any
            Tracking options, influencing the way to choose the next direction.
        """
        raise NotImplementedError

    def get_opposite_direction_sphere(self, ind):
        """
        Get the indice of the opposite direction on the sphere to the indice
        ind.

        Parameters
        ----------
        ind: int
            Indice of sphere direction.

        Return
        ------
        value: int
            Indice of opposite sphere direction.
        """
        return (len(self.dirs) // 2 + ind) % len(self.dirs)


class ODFField(AbstractTrackingField):
    """
    ODF (Spherical harmonics) tracking field.

    Parameters
    ----------
    odf_dataset: scilpy.image.datasets.DataVolume
        Trackable Dataset object.
    basis: string
        SH basis name. One of 'tournier07' or 'descoteaux07'
    sf_threshold: float
        Threshold on spherical function (SF).
    sf_threshold_init: float
        Threshold on spherical function when initializing a new streamline.
    theta: float
        Maximum angle (radians) between two steps.
    dipy_sphere: string, optional
        Name of the DIPY sphere object to use for evaluating SH. Can't be
        None.
    min_separation_angle: float, optional
        Minimum separation angle (in radians) for peaks extraction. Used for
        deterministic tracking. A candidate direction is a maximum if its SF
        value is greater than all other SF values in its neighbourhood, where
        the neighbourhood includes all the sphere directions located at most
        `min_separation_angle` from the candidate direction.
    """

    def __init__(self, odf_dataset, basis, sf_threshold, sf_threshold_init,
                 theta, dipy_sphere='symmetric724',
                 min_separation_angle=np.pi / 16.):
        super().__init__(odf_dataset, theta, dipy_sphere)

        self.sf_threshold = sf_threshold
        self.sf_threshold_init = sf_threshold_init
        sh_order = order_from_ncoef(self.dataset.data.shape[-1])
        self.basis = basis
        self.B = sh_to_sf_matrix(self.sphere, sh_order, self.basis,
                                 smooth=0.006, return_inv=False)

        # For deterministic tracking:
        self.maxima_neighbours = self._get_sphere_neighbours(
            min_separation_angle)

    def _get_sf(self, pos):
        """
        Get the spherical function at position pos.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in mm in the trackable dataset.

        Return
        ------
        sf: ndarray (len(self.sphere.vertices),)
            Spherical function evaluated at pos, normalized by
            its maximum amplitude.
        """
        sh = self.dataset.voxmm_to_value(*pos)
        sf = np.dot(self.B.T, sh).reshape((-1, 1))

        sf_max = np.max(sf)
        if sf_max > 0:
            sf = sf / sf_max
        return sf

    def get_possible_next_directions(self, pos, previous_direction,
                                     tracking_choice='prob'):
        """
        Get the set of next possible directions. If tracking_choice is 'prob',
        the next possible directions are all directions inside the cone of
        angle theta ahead (associated sf values are also returned to allow
        sampling). If tracking_choice is 'det', the set of possible directions
        are all maxima inside the cone theta. A point is considered a maxima if
        it has the greatest value in a neigbhorhood of angle
        min_separation_angle.

        The way to sample from these possible directions is the propagator's
        job.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset, expressed in mm.
        previous_direction: TrackingDirection
            Incoming direction. Outcoming direction won't be further than an
            angle theta.
        tracking_choice: str
            Either "prob" or "det"
        """
        if tracking_choice == 'prob':
            # Getting direction from the SF
            return self._get_possible_next_dirs_prob(pos, previous_direction)
        elif tracking_choice == 'det':
            # Getting direction from the maxima
            return self._get_possible_next_dirs_det(pos, previous_direction)
        else:
            raise ValueError("Tracking choice must be one of 'det' or 'prob'.")

    def _get_possible_next_dirs_prob(self, pos, previous_direction):
        """
        Get the spherical functions thresholded at position pos, for a given
        direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset, expressed in mm.
        previous_direction: TrackingDirection
            Incoming direction. Outcoming direction won't be further than an
            angle theta.

        Return
        ------
        value: tuple
            The neighbours SF evaluated at pos in given direction and
            corresponding tracking directions.
        """
        sf = self._get_sf(pos)
        sf[sf < self.sf_threshold] = 0
        inds = np.nonzero(
            self.tracking_neighbours[previous_direction.index])[0]
        return sf[inds], self.dirs[inds]

    def _get_possible_next_dirs_det(self, pos, previous_direction):
        """
        Get the set of maxima directions from the thresholded
        SF at position pos, for a direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset, expressed in mm.
        previous_direction: TrackingDirection
            Incoming direction. Outcoming direction won't be further than an
            angle theta.

        Return
        ------
        maxima: list
            List of directions of maxima around the input direction at pos.
        """
        sf = self._get_sf(pos)
        sf[sf < self.sf_threshold] = 0
        maxima = []
        for i in np.nonzero(self.tracking_neighbours[
                                previous_direction.index])[0]:
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
            Position in the dataset, expressed in mm.

        Return
        ------
        value: tuple
            Initial direction to follow from pos and its opposite direction.
        """
        sf = self._get_sf(pos)
        sf[sf < self.sf_threshold_init] = 0

        if np.sum(sf) > 0:
            ind = sample_distribution(sf)
            ind_opposite = self.get_opposite_direction_sphere(ind)
            return self.dirs[ind], self.dirs[ind_opposite]
        return None, None
