# -*- coding: utf-8 -*-
import numpy as np

from scilpy.tracking.tools import sample_distribution
from scilpy.tracking.utils import TrackingDirection
from scilpy.tracking.tracking_field import AbstractTrackingField, \
                                           ODFField


class AbstractPropagator(object):
    """
    Abstract class for tracker object.

    Parameters
    ----------
    tracking_field: scilpy tracking field object
        The TrackingField object on which the tracking is done.
    step_size: float
        The step size for tracking.
    rk_order: int
        Order for the Runge Kutta integration.
    """
    def __init__(self, tracking_field: AbstractTrackingField,
                 step_size, rk_order):
        self.tracking_field = tracking_field
        self.step_size = step_size
        if not (rk_order == 1 or rk_order == 2 or rk_order == 4):
            raise ValueError("Invalid runge-kutta order. Is " +
                             str(rk_order) + ". Choices : 1, 2, 4")
        self.rk_order = rk_order

        # Following values will be initialized when calling self.initialize
        self.init_pos = None
        self.forward_pos = None
        self.backward_pos = None
        self.forward_dir = None
        self.backward_dir = None

    def initialize(self, pos, track_forward_only):
        """
        Initialize the tracking at position pos. Initial tracking directions
        are picked, the propagete_foward() and propagate_backward() functions
        can then be call.

        Parameters
        ----------
        pos: ndarray (3,)
            Initial tracking position.
        track_forward_only: bool
            If true, verifies validity of forward direction only.

        Return
        ------
        value: bool
            True if initial tracking directions are found.
        """
        self.init_pos = pos
        self.forward_pos = pos
        self.backward_pos = pos
        self.forward_dir, self.backward_dir =\
            self.tracking_field.get_init_direction(pos)

        if track_forward_only:
            if self.forward_dir is not None:
                return True
            elif self.backward_dir is not None:
                self.forward_dir = self.backward_dir
                return True
            return False
        return self.forward_dir is not None and self.backward_dir is not None

    def _sample_next_direction_or_go_straight(self, pos, v_in):
        """
        Same as _sample_next_direction but if no valid direction has been
        found, return v_in as v_out.
        """
        is_direction_valid = True
        v_out = self._sample_next_direction(pos, v_in)
        if v_out is None:
            is_direction_valid = False
            v_out = v_in

        return is_direction_valid, v_out

    def propagate(self, pos, v_in):
        """
        Given the current position and direction, computes the next position
        and direction using Runge-Kutta integration method. If no valid
        tracking direction is available, v_in is chosen.

        Parameters
        ----------
        pos: ndarrray (3,)
            Current position.
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        ------
        new_pos: ndarray (3,)
            The new segment position.
        new_dir: ndarray (3,)
            The new segment direction.
        is_direction_valid: bool
            True if new_dir is valid.
        """
        if self.rk_order == 1:
            is_direction_valid, new_dir = \
                self._sample_next_direction_or_go_straight(pos, v_in)

        elif self.rk_order == 2:
            is_direction_valid, dir1 = \
                self._sample_next_direction_or_go_straight(pos, v_in)
            _, new_dir = self._sample_next_direction_or_go_straight(
                pos + 0.5 * self.step_size * np.array(dir1), dir1)

        else:
            # case self.rk_order == 4
            is_direction_valid, dir1 = \
                self._sample_next_direction_or_go_straight(pos, v_in)
            v1 = np.array(dir1)
            _, dir2 = self._sample_next_direction_or_go_straight(
                pos + 0.5 * self.step_size * v1, dir1)
            v2 = np.array(dir2)
            _, dir3 = self._sample_next_direction_or_go_straight(
                pos + 0.5 * self.step_size * v2, dir2)
            v3 = np.array(dir3)
            _, dir4 = self._sample_next_direction_or_go_straight(
                pos + self.step_size * v3, dir3)
            v4 = np.array(dir4)

            new_v = (v1 + 2 * v2 + 2 * v3 + v4) / 6
            new_dir = TrackingDirection(new_v, dir1.index)

        new_pos = pos + self.step_size * np.array(new_dir)

        return new_pos, new_dir, is_direction_valid

    def is_voxmm_in_bound(self, pos, origin='center'):
        """
        Test if the streamline point is inside the boundary of the image.

        Parameters
        ----------
        pos : tuple
            3D positions.
        origin: str
            'Center': Voxel 0,0,0 goes from [-resx/2, -resy/2, -resz/2] to
                [resx/2, resy/2, resz/2].
            'Corner': Voxel 0,0,0 goes from [0,0,0] to [resx, resy, resz].

        Return
        ------
        value: bool
            True if the streamline point is inside the boundary of the image.
        """
        return self.tracking_field.dataset.is_voxmm_in_bound(*pos, origin)

    def _sample_next_direction(self, pos, v_in):
        """
        Chooses a next tracking direction from all possible directions offered
        by the tracking field.
        Should use self.tracking_field.get_possible_next_directions.

        Parameters
        ----------
        pos: ndarray (3,)
            Current tracking position.
        v_in: ndarray (3,)
            Previous tracking direction.
        """
        pass


class ProbabilisticODFPropagator(AbstractPropagator):
    """
    Probabilistic direction tracker.

    Parameters
    ----------
    tracking_field: scilpy tracking field object
        The TrackingField object on which the tracking is done.
    step_size: float
        The step size for tracking.
    rk_order: int
        Order for the Runge Kutta integration.
    """
    def __init__(self, tracking_field: ODFField, step_size,
                 rk_order):
        super(ProbabilisticODFPropagator, self).__init__(
            tracking_field, step_size, rk_order)

    def _sample_next_direction(self, pos, v_in):
        """
        Chooses a next tracking direction from all possible directions offered
        by the tracking field.
        Should use self.tracking_field.get_next_direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Current tracking position.
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        ------
        direction: ndarray (3,)
            A valid tracking direction. None if no valid direction is found.
        """
        sf, directions = self.tracking_field.get_possible_next_directions(
            pos, v_in, 'prob')
        if np.sum(sf) > 0:
            return directions[sample_distribution(sf)]
        return None


class DeterministicODFPropagator(AbstractPropagator):
    """
    Deterministic direction tracker on the ODF. Direction is the maximum
    direction on the sphere.

    Parameters
    ----------
    tracking_field: scilpy tracking field object
        The TrackingField object on which the tracking is done.
    step_size: float
        The step size for tracking.
    rk_order: int
        Order for the Runge Kutta integration.
    """
    def __init__(self, tracking_field: ODFField, step_size,
                 rk_order):
        super(DeterministicODFPropagator, self).__init__(
            tracking_field, step_size, rk_order)

    def _sample_next_direction(self, pos, v_in):
        """
        Chooses a next tracking direction from all possible directions offered
        by the tracking field.
        Should use self.tracking_field.get_next_direction.
        Returns None if no valid maxima is available.

        Parameters
        ----------
        pos: ndarray (3,)
            Current tracking position.
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        ------
        direction: ndarray (3,)
            The maxima closest to v_in. None if the no
            valid maxima are available.
        """
        possible_maxima = self.tracking_field.get_possible_next_directions(
            pos, v_in, 'det')
        cosinus = 0
        v_out = None
        for d in possible_maxima:
            new_cosinus = np.dot(v_in, d)
            if new_cosinus > cosinus:
                cosinus = new_cosinus
                v_out = d
        return v_out
