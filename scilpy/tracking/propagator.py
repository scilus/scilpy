# -*- coding: utf-8 -*-
from enum import Enum
import logging

import numpy as np

import dipy
from dipy.io.stateful_tractogram import Space, Origin
from dipy.reconst.shm import sh_to_sf_matrix

from scilpy.reconst.utils import (get_sphere_neighbours,
                                  get_sh_order_and_fullness)
from scilpy.tracking.utils import sample_distribution, TrackingDirection


class PropagationStatus(Enum):
    ERROR = 1


class AbstractPropagator(object):
    """
    Abstract class for propagator object. "Propagation" means continuing the
    streamline a step further. The propagator is thus responsible for sampling
    the next direction at current step through Runge-Kutta integration
    (whereas the tracker using this propagator will be responsible for the
    processing parameters, number of streamlines, stopping criteria, etc.).

    Propagation depends on the type of data (ex, DTI, fODF) and the way to get
    a direction from it (ex, det, prob).
    """
    def __init__(self, datavolume, step_size, rk_order, space, origin):
        """
        Parameters
        ----------
        datavolume: scilpy.image.volume_space_management.DataVolume
            Trackable Dataset object.
        step_size: float
            The step size for tracking. Important: step size should be in the
            same units as the space of the tracking!
        rk_order: int
            Order for the Runge Kutta integration.
        space: dipy Space
            Space of the streamlines during tracking.
            value.
        origin: dipy Origin
            Origin of the streamlines during tracking. All coordinates received
            in the propagator's methods will be expected to respect
            that origin.

        A note on space and origin: All coordinates received in the
        propagator's methods will be expected to respect those values.
        Tracker will verify that the propagator has the same internal values as
        itself.
        """
        self.datavolume = datavolume

        self.origin = origin
        self.space = space

        # Propagation options
        self.step_size = step_size
        if not (rk_order == 1 or rk_order == 2 or rk_order == 4):
            raise ValueError("Invalid runge-kutta order. Is " +
                             str(rk_order) + ". Choices : 1, 2, 4")
        self.rk_order = rk_order

        # By default, normalizing directions. Adding option for child classes.
        self.normalize_directions = True

        # Will be reset at each new streamline.
        self.line_rng_generator = None

    def reset_data(self, new_data=None):
        """
        Reset data before starting a new process. In current implementation,
        we reset the internal data to None before starting a multiprocess, then
        load it back when process has started.

        Parameters
        ----------
        new_data: Any
            Will replace self.datavolume.data.

        """
        self.datavolume.data = new_data

    def prepare_forward(self, seeding_pos, random_generator):
        """
        Prepare information necessary at the first point of the
        streamline for forward propagation: v_in and any other information
        necessary for the self.propagate method.

        Parameters
        ----------
        seeding_pos: tuple(x,y,z)
            The seeding position. Important, position must be in the same space
            and origin as self.space, self.origin!
        random_generator: numpy Generator.

        Returns
        -------
        tracking_info: Any
            Any tracking information necessary for the propagation.
            Return PropagationStatus.ERROR if no good tracking direction can be
            set at current seeding position.
        """
        # To be defined by child classes.
        # Should set self.line_rng_generator = random_generator
        raise NotImplementedError

    def prepare_backward(self, line, forward_dir):
        """
        Called at the beginning of backward tracking, in case we need to
        reset some parameters

        Parameters
        ----------
        line: List
            Result from the forward tracking, reversed.
        forward_dir: ndarray (3,)
            v_in chosen at the forward step.

        Returns
        -------
        v_in: ndarray (3,)
            Last direction of the streamline. If the streamline contains
            only the seeding point (forward tracking failed), simply inverse
            the forward direction.
        """
        if len(line) > 1:
            v = line[-1] - line[-2]
            if self.normalize_directions:
                return v / np.linalg.norm(v)
            else:
                return v
        elif forward_dir is not None:
            return [-dir_i for dir_i in forward_dir]
        else:
            return None

    def finalize_streamline(self, last_pos, v_in):
        """
        Return the last position of the streamline.

        Parameters
        ----------
        last_pos: ndarray (3,)
            Last propagated position. Important, position must be in the same
            space and origin as self.space, self.origin!
        v_in: TrackingDirection
            Last propagated direction.

        Returns
        -------
        final_pos: ndarray (3,)
            Position of the final point of the streamline. Return None, or
            last_pos, if no last step is wished.
        """
        # Make a last step straight in the last direction (no sampling or
        # interpolation of a new direction). Ex of use: if stopped because it
        # exited the (WM) tracking mask, reaching GM a little more.
        final_pos = last_pos + self.step_size * np.array(v_in)
        return final_pos

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

    def propagate(self, line, v_in):
        """
        Given the current position and direction, computes the next position
        and direction using Runge-Kutta integration method. If no valid
        tracking direction is available, v_in is chosen.

        Parameters
        ----------
        line: list[ndarrray (3,)]
            Current position.
        v_in: ndarray (3,) or TrackingDirection
            Previous tracking direction.

        Return
        ------
        new_pos: ndarray (3,)
            The new segment position, expressed in propagator's space and
            origin.
        new_dir: ndarray (3,) or TrackingDirection
            The new segment direction.
        is_direction_valid: bool
            True if new_dir is valid.
        """
        # Finding last coordinate
        pos = line[-1]

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

    def _sample_next_direction(self, pos, v_in):
        """
        Chooses a next tracking direction from all possible directions offered
        by the tracking field.

        Parameters
        ----------
        pos: ndarray (3,)
            Current tracking position.  Important, position must be in the same
            space and origin as self.space, self.origin!
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        -------
        direction: ndarray (3,)
            A valid tracking direction. None if no valid direction is found.
            Direction should be normalized.
        """
        raise NotImplementedError


class PropagatorOnSphere(AbstractPropagator):
    def __init__(self, datavolume, step_size, rk_order, dipy_sphere,
                 sub_sphere, space, origin):
        """
        Parameters
        ----------
        datavolume: scilpy.image.volume_space_management.DataVolume
            Trackable DataVolume object.
        step_size: float
            The step size for tracking.
        rk_order: int
            Order for the Runge Kutta integration.
        dipy_sphere: string, optional
            If necessary, name of the DIPY sphere object to use to evaluate
            directions.
        space: dipy Space
            Space of the streamlines during tracking.
        origin: dipy Origin
            Origin of the streamlines during tracking.
        """
        super().__init__(datavolume, step_size, rk_order, space, origin)

        self.sphere = dipy.data.get_sphere(dipy_sphere).subdivide(sub_sphere)
        self.dirs = np.zeros(len(self.sphere.vertices), dtype=np.ndarray)
        for i in range(len(self.sphere.vertices)):
            self.dirs[i] = TrackingDirection(self.sphere.vertices[i], i)

    def prepare_backward(self, line, forward_dir):
        """
        Called at the beginning of backward tracking, in case we need to
        reset some parameters

        Parameters
        ----------
        line: List
            Result from the forward tracking, reversed.
        forward_dir: ndarray (3,)
            v_in chosen at the forward step.

        Returns
        -------
        v_in: ndarray (3,)
            Last direction of the streamline, of if it contains only the
            seeding point (forward tracking failed), simply inverse the
            forward direction.
        """
        if len(line) > 1:
            last_dir = line[-1] - line[-2]
            ind = self.sphere.find_closest(last_dir)
        else:
            backward_dir = -np.asarray(forward_dir)
            ind = self.sphere.find_closest(backward_dir)

        # toDo. Is using a TrackingDirection necessary compared to a direction
        #  x,y, z or rho, phi? self.sphere.vertices[ind] might not be
        #  exactly equal to last_dir or to backward_dir.
        return TrackingDirection(self.sphere.vertices[ind], ind)


class ODFPropagator(PropagatorOnSphere):
    """
    Propagator on ODFs/fODFs. Algo can be det or prob.
    """
    def __init__(self, datavolume, step_size,
                 rk_order, algo, basis, sf_threshold, sf_threshold_init,
                 theta, dipy_sphere='symmetric724',
                 sub_sphere=0,
                 min_separation_angle=np.pi / 16.,
                 space=Space('vox'), origin=Origin('center'),
                 is_legacy=True):
        """

        Parameters
        ----------
        datavolume: scilpy.image.volume_space_management.DataVolume
            Trackable DataVolume object.
        step_size: float
            The step size for tracking.
        rk_order: int
            Order for the Runge Kutta integration.
        algo: string
            Type of algorithm. Choices are 'det' or 'prob'
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
        sub_sphere: int
            Number of subdivisions to use for the sphere.
        min_separation_angle: float, optional
            Minimum separation angle (in radians) for peaks extraction. Used
            for deterministic tracking. A candidate direction is a maximum if
            its SF value is greater than all other SF values in its
            neighbourhood, where the neighbourhood includes all the sphere
            directions located at most `min_separation_angle` from the
            candidate direction.
        space: dipy Space
            Space of the streamlines during tracking. Default: VOX, like in
            dipy. Interpolation of the ODF is done in VOX space (see
            DataVolume.vox_to_value) so this choice implies the less data
            modification.
        origin: dipy Origin
            Origin of the streamlines during tracking. Default: center, like in
            dipy. Interpolation of the ODF is done in center origin so this
            choice implies the less data modification.
        is_legacy : bool, optional
            Whether or not the SH basis is in its legacy form.
        """
        super().__init__(datavolume, step_size, rk_order, dipy_sphere,
                         sub_sphere, space, origin)

        if self.space == Space.RASMM:
            raise NotImplementedError(
                "This version of the propagator on ODF is not ready to work "
                "in RASMM space.")

        # Warn user if the rk order does not match the algo
        if rk_order != 1 and algo == 'prob':
            logging.warning('Probabilistic tracking with RK order != 1 is '
                            'not recommended! Use deterministic tracking '
                            'or set rk_order to 1 instead.')

        # Propagation params
        self.theta = theta
        if algo not in ['det', 'prob']:
            raise ValueError("ODFPropagator algo should be 'det' or 'prob'.")
        self.algo = algo
        self.tracking_neighbours = get_sphere_neighbours(self.sphere,
                                                         self.theta)
        # For deterministic tracking:
        self.maxima_neighbours = get_sphere_neighbours(self.sphere,
                                                       min_separation_angle)

        # ODF params
        self.sf_threshold = sf_threshold
        self.sf_threshold_init = sf_threshold_init
        sh_order, full_basis =\
            get_sh_order_and_fullness(self.datavolume.data.shape[-1])
        self.basis = basis
        self.is_legacy = is_legacy
        self.B = sh_to_sf_matrix(self.sphere, sh_order, self.basis,
                                 smooth=0.006, return_inv=False,
                                 full_basis=full_basis, legacy=self.is_legacy)

    def _get_sf(self, pos):
        """
        Get the spherical function at position pos.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in the trackable dataset. Important, position should be
            in the same space and origin as self.space, self.origin!

        Return
        ------
        sf: ndarray (len(self.sphere.vertices),)
            Spherical function evaluated at pos, normalized by
            its maximum amplitude.
        """
        # Interpolation:
        sh = self.datavolume.get_value_at_coordinate(
            *pos, space=self.space, origin=self.origin)
        sf = np.dot(self.B.T, sh).reshape((-1, 1))

        sf_max = np.max(sf)
        if sf_max > 0:
            sf /= sf_max
        return sf

    def prepare_forward(self, seeding_pos, random_generator):
        """
        Prepare information necessary at the first point of the
        streamline for forward propagation: v_in and any other information
        necessary for the self.propagate method.

        About **v_in**, it is used for two things:

        - To sample the next direction based on _sample_next_direction method.
            Ex, with fODF, it defines a cone theta of accepable directions.
        - If no valid next dir are found, continue straight.

        Parameters
        ----------
        seeding_pos: tuple(x,y,z)
            The seeding position. Important, position must be in the same space
            and origin as self.space, self.origin!
        random_generator: numpy Generator

        Returns
        -------
        v_in: TrackingDirection
            The "fake" previous direction at first step. Could be None if your
            propagator can propagate without knowledge of previous direction.
            Return PropagationStatus.Error if no good tracking direction can be
            set at current seeding position.
        """
        # Sampling on the SF values (no matter if general algo is det or prob)
        # with a different threshold than usual (sf_threshold_init).
        # So the initial step's propagation will be in a cone theta around a
        # "more probable" peak.
        sf = self._get_sf(seeding_pos)
        sf[sf < self.sf_threshold_init] = 0
        self.line_rng_generator = random_generator

        if np.sum(sf) > 0:
            ind = sample_distribution(sf, self.line_rng_generator)
            return TrackingDirection(self.dirs[ind], ind)

        # Else: sf at current position is smaller than acceptable threshold in
        # all directions.
        return PropagationStatus.ERROR

    def _sample_next_direction(self, pos, v_in):
        """
        Chooses a next tracking direction from all possible directions offered
        by the tracking field.

        Parameters
        ----------
        pos: ndarray (3,)
            Current tracking position.  Important, position must be in the same
            space and origin as self.space, self.origin!
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        ------
        direction: ndarray (3,)
            A valid tracking direction. None if no valid direction is found.
        """
        if self.algo == 'prob':
            # Tracking field returns the sf and directions
            sf, directions = self._get_possible_next_dirs_prob(pos, v_in)

            # Sampling one.
            if np.sum(sf) > 0:
                v_out = directions[sample_distribution(sf,
                                                       self.line_rng_generator)]
            else:
                return None
        elif self.algo == 'det':
            # Tracking field returns the list of possible maxima.
            possible_maxima = self._get_possible_next_dirs_det(pos, v_in)
            # Choosing one.
            cosinus = 0
            v_out = None
            for d in possible_maxima:
                new_cosinus = np.dot(v_in, d)
                if new_cosinus > cosinus:
                    cosinus = new_cosinus
                    v_out = d
        else:
            raise ValueError("Tracking choice must be one of 'det' or 'prob'.")

        # Not normalizing: direction comes from dipy's (unit) sphere so
        # supposing that it's ok.
        return v_out

    def _get_possible_next_dirs_prob(self, pos, v_in):
        """
        Get the spherical functions thresholded at position pos, for a given
        direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset. Important, position must be in the
            same space and origin as self.space, self.origin!
        v_in: TrackingDirection
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
            self.tracking_neighbours[v_in.index])[0]
        return sf[inds], self.dirs[inds]

    def _get_possible_next_dirs_det(self, pos, previous_direction):
        """
        Get the set of maxima directions from the thresholded
        SF at position pos, for a direction.

        Parameters
        ----------
        pos: ndarray (3,)
            Position in trackable dataset. Important, position must be in the
            same space and origin as self.space, self.origin!
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
