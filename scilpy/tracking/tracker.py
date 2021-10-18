# -*- coding: utf-8 -*-
import numpy as np

from scilpy.tracking.tools import sample_distribution
from scilpy.tracking.utils import TrackingDirection
from scilpy.tracking.tracking_field import AbstractTrackingField, \
                                           SphericalHarmonicField


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

    def initialize(self, pos):
        """
        Initialize the tracking at position pos. Initial tracking directions
        are picked, the propagete_foward() and propagate_backward() functions
        can then be call.

        Parameters
        ----------
        pos: ndarray (3,)
            Initial tracking position.

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
        return self.forward_dir is not None and self.backward_dir is not None

    def get_next_valid_direction(self, pos, v_in):
        """
        Get the next direction given the position pos, input direction
        v_in, and tracking method (ex, probabilistic or deterministic), and
        verify if it is valid. If it is not valid, return v_in as next
        direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Current 3D position.
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        ------
        is_direction_valid: bool
            True if the new direction is valid.
        v_out: ndarray(3,)
            A valid direction. v_out equals v_in if no valid direction is
            found.
        """
        is_direction_valid = True
        v_out = self.get_direction(pos, v_in)
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
            is_direction_valid, new_dir = self.get_next_valid_direction(
                pos, v_in)

        elif self.rk_order == 2:
            is_direction_valid, dir1 = self.get_next_valid_direction(
                pos, v_in)
            _, new_dir = self.get_next_valid_direction(
                pos + 0.5 * self.step_size * np.array(dir1), dir1)

        else:
            # case self.rk_order == 4
            is_direction_valid, dir1 = self.get_next_valid_direction(
                pos, v_in)
            v1 = np.array(dir1)
            _, dir2 = self.get_next_valid_direction(
                pos + 0.5 * self.step_size * v1, dir1)
            v2 = np.array(dir2)
            _, dir3 = self.get_next_valid_direction(
                pos + 0.5 * self.step_size * v2, dir2)
            v3 = np.array(dir3)
            _, dir4 = self.get_next_valid_direction(
                pos + self.step_size * v3, dir3)
            v4 = np.array(dir4)

            new_v = (v1 + 2 * v2 + 2 * v3 + v4) / 6
            new_dir = TrackingDirection(new_v, dir1.index)

        new_pos = pos + self.step_size * np.array(new_dir)

        return new_pos, new_dir, is_direction_valid

    def is_position_in_bound(self, pos):
        """
        Test if the streamline point is inside the boundary of the image.

        Parameters
        ----------
        pos : tuple
            3D positions.

        Return
        ------
        value: bool
            True if the streamline point is inside the boundary of the image.
        """
        return self.tracking_field.dataset.is_position_in_bound(*pos)

    def get_direction(self, pos, v_in):
        """
        Abstract method. Return the next tracking direction, given
        the current position pos and the previous direction v_in.
        This direction must respect tracking constraint defined in
        the tracking_field.

        Parameters
        ----------
        pos: ndarray (3,)
            Current tracking position.
        v_in: ndarray (3,)
            Previous tracking direction.
        """
        pass


class ProbabilisticSHPropagator(AbstractPropagator):
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
    def __init__(self, tracking_field: SphericalHarmonicField, step_size,
                 rk_order):
        super(ProbabilisticSHPropagator, self).__init__(
            tracking_field, step_size, rk_order)

    def get_direction(self, pos, v_in):
        """
        Return the next tracking direction, given the current position
        pos and the previous direction v_in. This direction must respect
        tracking constraint defined in the tracking_field.

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
        sf, directions = self.tracking_field.get_next_direction(
            pos, v_in, 'prob')
        if np.sum(sf) > 0:
            return directions[sample_distribution(sf)]
        return None


class DeterministicMaximaSHPropagator(AbstractPropagator):
    """
    Deterministic direction tracker.

    Parameters
    ----------
    tracking_field: scilpy tracking field object
        The TrackingField object on which the tracking is done.
    step_size: float
        The step size for tracking.
    rk_order: int
        Order for the Runge Kutta integration.
    """
    def __init__(self, tracking_field: SphericalHarmonicField, step_size,
                 rk_order):
        super(DeterministicMaximaSHPropagator, self).__init__(
            tracking_field, step_size, rk_order)

    def get_direction(self, pos, v_in):
        """
        Get the next valid tracking direction or None if no valid maxima
        is available.

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
        possible_maxima = self.tracking_field.get_next_direction(
            pos, v_in, 'det')
        cosinus = 0
        v_out = None
        for d in possible_maxima:
            new_cosinus = np.dot(v_in, d)
            if new_cosinus > cosinus:
                cosinus = new_cosinus
                v_out = d
        return v_out
