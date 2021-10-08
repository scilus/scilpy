# -*- coding: utf-8 -*-
import dipy.data
from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
import numpy as np

from scilpy.tracking.tools import sample_distribution
from scilpy.tracking.utils import TrackingDirection


class AbstractTrackingField(object):
    def __init__(self, dataset, theta, dipy_sphere='symmetric724'):
        """
        dataset: Any
            The type of dataset and the way to use it will depend on the child
            implementation.
        theta: float
            Maximum angle (radians) between two steps.
        dipy_sphere: string
            Name of the DIPY sphere object to use to decide which neighbors to
            evaluate.
        """
        self.dataset = dataset
        self.theta = theta

        if 'symmetric' not in dipy_sphere:
            raise ValueError('Sphere must be symmetric. Call to '
                             'get_opposite_direction will fail.')
        self.sphere = dipy.data.get_sphere(dipy_sphere)

        # Instantiating the neighbors:
        self.tracking_neighbours = self.get_direction_neighbours(self.theta)

        # Instantiating tracking direction list:
        # (A 3D cartesian direction (list(x,y,z)) + an index to work with
        # discrete sphere).
        self.dirs = np.zeros(len(self.sphere.vertices), dtype=np.ndarray)
        for i in range(len(self.sphere.vertices)):
            self.dirs[i] = TrackingDirection(self.sphere.vertices[i], i)

    def get_direction_neighbours(self, max_angle):
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
        xs = self.sphere.vertices[:, 0]
        ys = self.sphere.vertices[:, 1]
        zs = self.sphere.vertices[:, 2]
        scalar_prods = np.outer(xs, xs) + np.outer(ys, ys) + np.outer(zs, zs)
        neighbours = scalar_prods >= np.cos(max_angle)
        return neighbours

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

    def get_init_direction(self, pos):
        """
        Get a tuple with an initial direction to follow from position pos,
        with the opposite direction.

        Must be instantiated by each child class.
        """
        raise NotImplementedError

    def get_next_direction(self, pos, direction, options=None):
        """
        Get a tuple with a next direction to follow from position pos,
        with the opposite direction.

        Must be instantiated by each child class.
        """
        raise NotImplementedError


class SphericalHarmonicField(AbstractTrackingField):
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
    angle_maxima_detection: float
        Angle used for peak extraction to check if a direction is maximal in
        its neighbourhood.
    """

    def __init__(self, odf_dataset, basis, sf_threshold, sf_threshold_init,
                 theta, dipy_sphere='symmetric724',
                 angle_maxima_detection=np.pi / 16.):
        super().__init__(odf_dataset, theta, dipy_sphere)

        self.sf_threshold = sf_threshold
        self.sf_threshold_init = sf_threshold_init

        self.maxima_neighbours = self.get_direction_neighbours(
            angle_maxima_detection)
        self.basis = basis
        sh_order = order_from_ncoef(self.dataset.data.shape[-1])
        self.B = sh_to_sf_matrix(self.sphere, sh_order, self.basis,
                                 smooth=0.006, return_inv=False)

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
        sh = self.dataset.getPositionValue(*pos)
        sf = np.dot(self.B.T, sh).reshape((-1, 1))

        sf_max = np.max(sf)
        if sf_max > 0:
            sf = sf / sf_max
        return sf

    def _get_tracking_sf(self, pos, direction):
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

    def _get_tracking_maxima(self, pos, direction):
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

    def get_next_direction(self, pos, direction, option):
        """
        Get the set of next directions, for a direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset, expressed in mm.
        direction: TrackingDirection
            A given direction.
        option: str
            Either "from_SF" or "from_maxima"
        """
        if option == 'from_SF':
            # Getting direction from the SF
            return self._get_tracking_sf(pos, direction)
        if option == 'from_maxima':
            # Getting direction from the maxima
            return self._get_tracking_maxima(pos, direction)
