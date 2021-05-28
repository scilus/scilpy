
from __future__ import division

import numpy as np

import scilpy.tracking.tools
import scilpy.tracking.tracking_field_bitbucket


class abstractPropagator(object):

    def __init__(self, tracker, param):
        self.step_size = param['step_size']
        self.tracker = tracker

    def propagate(self, pos, v_in):
        scilpy.utils.abstract()

    def getValidDirection(self, pos, v_in):
        is_valid_direction = True
        v_out = self.tracker.get_direction(pos, v_in)
        if v_out is None:
            is_valid_direction = False
            v_out = v_in

        return is_valid_direction, v_out


class rk1Propagator(abstractPropagator):

    """
    The order 1 Runge Kutta propagator is equivalent to the step function
    used before the implementation of the Runge Kutta integration
    """
    def __init__(self, tracker, param):
        super(rk1Propagator, self).__init__(tracker, param)

    def propagate(self, pos, v_in):
        is_valid_direction, newDir = self.getValidDirection(pos, v_in)
        newPos = pos + self.step_size * np.array(newDir)
        return newPos, newDir, is_valid_direction


class rk2Propagator(abstractPropagator):

    def __init__(self, tracker, param):
        super(rk2Propagator, self).__init__(tracker, param)

    def propagate(self, pos, v_in):
        is_valid_direction, dir1 = self.getValidDirection(pos, v_in)
        newDir = self.getValidDirection(
            pos + 0.5 * self.step_size * np.array(dir1), dir1)[1]
        newPos = pos + self.step_size * np.array(newDir)
        return newPos, newDir, is_valid_direction


class rk4Propagator(abstractPropagator):

    def __init__(self, tracker, param):
        super(rk4Propagator, self).__init__(tracker, param)

    def propagate(self, pos, v_in):
        is_valid_direction, dir1 = self.getValidDirection(pos, v_in)
        v1 = np.array(dir1)
        dir2 = self.getValidDirection(pos + 0.5 * self.step_size * v1, dir1)[1]
        v2 = np.array(dir2)
        dir3 = self.getValidDirection(pos + 0.5 * self.step_size * v2, dir2)[1]
        v3 = np.array(dir3)
        dir4 = self.getValidDirection(pos + self.step_size * v3, dir3)[1]
        v4 = np.array(dir4)

        newV = (v1 + 2 * v2 + 2 * v3 + v4) / 6
        newDir = scilpy.tracking.trackingField.TrackingDirection(
            newV, dir1.index)
        newPos = pos + self.step_size * newV

        return newPos, newDir, is_valid_direction


class abstractTracker(object):

    def __init__(self, trackingField, param):
        self.trackingField = trackingField
        self.step_size = param['step_size']
        if param['rk_order'] == 1:
            self.propagator = rk1Propagator(self, param)
        elif param['rk_order'] == 2:
            self.propagator = rk2Propagator(self, param)
        elif param['rk_order'] == 4:
            self.propagator = rk4Propagator(self, param)
        else:
            raise ValueError("Invalid runge-kutta order. Is " +
                             str(param['rk_order']) + ". Choices : 1, 2, 4")

    def initialize(self, pos):
        """
        Initialise the tracking at position pos. Initial tracking directions
        are picked, the propagete_foward() and propagate_backward() functions
        could then be call.
        return True if initial tracking directions are found.
        """
        self.init_pos = pos
        self.forward_pos = pos
        self.backward_pos = pos
        self.forward_dir, self.backward_dir =\
            self.trackingField.get_init_direction(pos)
        return self.forward_dir is not None and self.backward_dir is not None

    def propagate(self, pos, v_in):
        """
        return tuple. The new tracking direction and the updated position.
        If no valid tracking direction are available, v_in is choosen.
        """
        return self.propagator.propagate(pos, v_in)

    def isPositionInBound(self, pos):
        """
        Test if the streamline point is inside the boundary of the image.

        Parameters
        ----------
        pos : tuple, 3D positions.

        Returns
        -------
        boolean
        """
        return self.trackingField.dataset.isPositionInBound(*pos)

    def get_direction(self, pos, v_in):
        """
        return the next tracking direction, given the current position pos
        and the previous direction v_in. This direction must respect tracking
        constraint defined in the trackingField.
        """
        scilpy.utils.abstract()


class probabilisticTracker(abstractTracker):

    def __init__(self, abstractDiscreteField, param):
        super(probabilisticTracker, self).__init__(
            abstractDiscreteField, param)

    def get_direction(self, pos, v_in):
        """
        return a direction drawn from the distribution weighted with
        the spherical function.
        None if the no valid direction are available.
        """
        sf, directions = self.trackingField.get_tracking_SF(pos, v_in)
        if np.sum(sf) > 0:
            return directions[scilpy.tracking.tools.sample_distribution(sf)]
        return None


class deterministicMaximaTracker(abstractTracker):

    def __init__(self, trackingField, param):
        super(deterministicMaximaTracker, self).__init__(
            trackingField, param)

    def get_direction(self, pos, v_in):
        """
        return the maxima the closest to v_in.
        None if the no valid maxima are available.
        """
        maxima_direction = self.trackingField.get_tracking_maxima(pos, v_in)
        cosinus = 0
        v_out = None
        for d in maxima_direction:
            new_cosinus = np.dot(v_in, d)
            if new_cosinus > cosinus:
                cosinus = new_cosinus
                v_out = d
        return v_out


class deterministicTENDPeakTracker(abstractTracker):

    def __init__(self, trackingField, param, g_param, fa):
        super(deterministicMaximaTracker, self).__init__(trackingField, param)
        self.g = g_param
        self.fa = fa

    def get_direction(self, pos, v_in):
        """
        See ISMRM 2013 Maxime Chamberland and Maxime Descoteaux.
        V = f * V_out + (1-f) * ((1-g) * V_in + g * V_out)
        """
        v_in = np.array(v_in)
        maxima_direction = self.trackingField.get_maxima(pos)
        min_cos = self.trackingField.cos_theta
        v_out = None
        for d in maxima_direction:
            new_cos = np.dot(v_in, d)
            if new_cos > min_cos:
                min_cos = new_cos
                v_out = d
        if v_out is None:
            return None
        v_out = np.array(v_out)
        f = self.fa.getPositionValue(*pos)
        return f * v_out + (1.0 - f) * ((1.0 - self.g) * v_in + self.g * v_out)
