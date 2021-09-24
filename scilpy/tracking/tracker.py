# -*- coding: utf-8 -*-
import numpy as np

from scilpy.tracking.tools import sample_distribution
from scilpy.tracking.utils import TrackingDirection


class abstractPropagator(object):
    """
    Base class for propagator objects.

    Parameters
    ----------
    tracker: scilpy tracker object
        The tracker the use for the propagator.
    step_size: float
        The step size used for tracking.
    """
    def __init__(self, tracker, step_size):
        self.step_size = step_size
        self.tracker = tracker

    def propagate(self, pos, v_in):
        """
        Abstract propagation method. Given the current position and
        direction, computes the next position and direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Current 3D position.
        v_in: ndarray (3,)
            Previous tracking direction.
        """
        pass

    def getValidDirection(self, pos, v_in):
        """
        Get the next valid direction given the position pos and
        input direction v_in.

        Parameters
        ----------
        pos: ndarray (3,)
            Current 3D position.
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        ------
        is_valid_direction: bool
            True if the new direction is valid.
        v_out: ndarray(3,)
            A valid direction. v_out equals v_in if no valid
            direction is found.
        """
        is_valid_direction = True
        v_out = self.tracker.get_direction(pos, v_in)
        if v_out is None:
            is_valid_direction = False
            v_out = v_in

        return is_valid_direction, v_out


class rk1Propagator(abstractPropagator):
    """
    Implementation of the order 1 Runge Kutta integration, equivalent to
    the Euler integration method.

    Parameters
    ----------
    tracker: scilpy tracker object
        The tracker the use for the propagator.
    step_size: float
        The step size used for tracking.
    """
    def __init__(self, tracker, step_size):
        super(rk1Propagator, self).__init__(tracker, step_size)

    def propagate(self, pos, v_in):
        """
        Given the current position and direction, computes the next position
        and direction using the RK1 integration method.

        Parameters
        ----------
        pos: ndarray (3,)
            Current 3D position.
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        ------
        new_pos: ndarray (3,)
            The new segment position.
        new_dir: ndarray (3,)
            The new segment direction.
        is_valid_direction: bool
            True if new_dir is valid.
        """
        is_valid_direction, new_dir = self.getValidDirection(pos, v_in)
        new_pos = pos + self.step_size * np.array(new_dir)
        return new_pos, new_dir, is_valid_direction


class rk2Propagator(abstractPropagator):
    """
    Implementation of the Runge Kutta integration method of order 2.

    Parameters
    ----------
    tracker: scilpy tracker object
        The tracker the use for the propagator.
    step_size: float
        The step size used for tracking.
    """
    def __init__(self, tracker, step_size):
        super(rk2Propagator, self).__init__(tracker, step_size)

    def propagate(self, pos, v_in):
        """
        Given the current position and direction, computes the next position
        and direction using the RK2 integration method.

        Parameters
        ----------
        pos: ndarray (3,)
            Current 3D position.
        v_in: ndarray (3,)
            Pervious tracking direction.

        Return
        ------
        new_pos: ndarray (3,)
            The new segment position.
        new_dir: ndarray (3,)
            The new segment direction.
        is_valid_direction: bool
            True if new_dir is valid.
        """
        is_valid_direction, dir1 = self.getValidDirection(pos, v_in)
        newDir = self.getValidDirection(
            pos + 0.5 * self.step_size * np.array(dir1), dir1)[1]
        newPos = pos + self.step_size * np.array(newDir)
        return newPos, newDir, is_valid_direction


class rk4Propagator(abstractPropagator):
    """
    Implementation of the Runge Kutta integration method of order 4.

    Parameters
    ----------
    tracker: scilpy tracker object
        The tracker the use for the propagator.
    step_size: float
        The step size used for tracking.
    """
    def __init__(self, tracker, step_size):
        super(rk4Propagator, self).__init__(tracker, step_size)

    def propagate(self, pos, v_in):
        """
        Given the current position and direction, computes the next position
        and direction using the RK4 integration method.

        Parameters
        ----------
        pos: ndarray (3,)
            Current 3D position.
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        ------
        new_pos: ndarray (3,)
            The new segment position.
        new_dir: ndarray (3,)
            The new segment direction.
        is_valid_direction: bool
            True if new_dir is valid.
        """
        is_valid_direction, dir1 = self.getValidDirection(pos, v_in)
        v1 = np.array(dir1)
        dir2 = self.getValidDirection(pos + 0.5 * self.step_size * v1, dir1)[1]
        v2 = np.array(dir2)
        dir3 = self.getValidDirection(pos + 0.5 * self.step_size * v2, dir2)[1]
        v3 = np.array(dir3)
        dir4 = self.getValidDirection(pos + self.step_size * v3, dir3)[1]
        v4 = np.array(dir4)

        newV = (v1 + 2 * v2 + 2 * v3 + v4) / 6
        newDir = TrackingDirection(newV, dir1.index)
        newPos = pos + self.step_size * newV

        return newPos, newDir, is_valid_direction


class abstractTracker(object):
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
    def __init__(self, tracking_field, step_size, rk_order):
        self.tracking_field = tracking_field
        self.step_size = step_size
        if rk_order == 1:
            self.propagator = rk1Propagator(self, step_size)
        elif rk_order == 2:
            self.propagator = rk2Propagator(self, step_size)
        elif rk_order == 4:
            self.propagator = rk4Propagator(self, step_size)
        else:
            raise ValueError("Invalid runge-kutta order. Is " +
                             str(rk_order) + ". Choices : 1, 2, 4")

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

    def propagate(self, pos, v_in):
        """
        Propagate a streamline. The new tracking direction and the
        updated position. If no valid tracking direction are available,
        v_in is choosen.

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
        is_valid_direction: bool
            True if new_dir is valid.
        """
        return self.propagator.propagate(pos, v_in)

    def isPositionInBound(self, pos):
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
        return self.tracking_field.dataset.isPositionInBound(*pos)

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

    def get_all_directions(self, pos, v_in):
        """
        return all maxima at pos.
        None if the no valid maxima are available.
        """
        maxima_direction = self.tracking_field.get_tracking_maxima(pos, v_in)
        v_out = []
        for d in maxima_direction:
            v_out.append(d)
        return v_out


class probabilisticTracker(abstractTracker):
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
    def __init__(self, tracking_field, step_size, rk_order):
        super(probabilisticTracker, self).__init__(
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
        sf, directions = self.tracking_field.get_tracking_SF(pos, v_in)
        if np.sum(sf) > 0:
            return directions[sample_distribution(sf)]
        return None


class deterministicMaximaTracker(abstractTracker):
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
    def __init__(self, tracking_field, step_size, rk_order):
        super(deterministicMaximaTracker, self).__init__(
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
        maxima_direction = self.tracking_field.get_tracking_maxima(pos, v_in)
        cosinus = 0
        v_out = None
        for d in maxima_direction:
            new_cosinus = np.dot(v_in, d)
            if new_cosinus > cosinus:
                cosinus = new_cosinus
                v_out = d
        return v_out
