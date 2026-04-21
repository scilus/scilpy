# -*- coding: utf-8 -*-

import logging
import numpy as np


class RAP:
    def __init__(self, rap_volume, propagator, max_nbr_pts):
        """
        rap_volume: DataVolume
            HRegion-Adaptive Propagation tractography volume.
        """
        self.rap_volume = rap_volume
        self.propagator = propagator
        self.max_nbr_pts = max_nbr_pts
        self._current_label = None
        self._total_steps = 0

    def is_in_rap_region(self, curr_pos, space, origin):
        return self.rap_volume.get_value_at_coordinate(
            *curr_pos, space=space, origin=origin) > 0

    def rap_multistep_propagate(self, line, prev_direction):
        """
        All child classes must implement this method. Must receive and return
        the parameters as defined here:


        Params
        ------
        line: list
            The beginning of the streamline

        Returns
        -------
        line: list
            The streamline extended with RAP in the RAP neighborhood.
        prev_direction: tuple
            The last direction (x, y, z).
        is_line_valid: bool
            If the line generated with RAP is valid.
        """
        raise NotImplementedError


class RAPContinue(RAP):
    """Dummy RAP class for tests. Goes straight"""

    def __init__(self, rap_volume, propagator, max_nbr_pts, step_size):
        """
        Step size: float
            The step size inside the RAP mask. Could be different from the step
            size elsewhere. In voxel world.
        """
        super().__init__(rap_volume, propagator, max_nbr_pts)
        self.step_size = step_size

    def rap_multistep_propagate(self, line, prev_direction):
        is_line_valid = True
        if len(line) > 3:
            pos = line[-2] + self.step_size * np.array(prev_direction)
            line[-1] = pos
            return line, prev_direction, is_line_valid
        return line, prev_direction, is_line_valid


class RAPSwitch(RAP):
    """RAP class that switches tracking parameters when inside the RAP mask/label."""

    def __init__(self, rap_volume, propagators: dict,
                 max_nbr_pts):
        """
        Parameters
        ----------
        rap_volume : DataVolume
            Region-Adaptive Propagation mask.
        propagators : dict
            Dictionary of ODFPropagator instances keyed by label (str).
            If --in_odf is provided, contains {odf_path: propagator}
            as default. Additional propagators are keyed by their label,
            loaded from the 'filename' key in rap_policies.json.
        max_nbr_pts : int
            Maximum number of points per streamline.
        """
        base_propagator = list(propagators.values())[
            0] if propagators else None
        super().__init__(rap_volume, base_propagator, max_nbr_pts)
        self._propagators = propagators

        if self.propagator is not None:
            self._base = {
                'step_size': self.propagator.step_size,
                'theta': self.propagator.theta,
                'algo': getattr(self.propagator, 'algo', None),
                'tracking_neighbours': getattr(self.propagator,
                                               'tracking_neighbours', None)
            }
        else:
            self._base = {}

        # Check if all labels in the volume are covered by the configuration
        unique_labels = np.unique(rap_volume.data)
        # Remove 0 (background) and convert to int
        unique_labels = [int(label) for label in unique_labels if label > 0]

        if unique_labels:
            missing_labels = [label for label in unique_labels
                              if str(label) not in self._propagators.keys()]
            if missing_labels:
                logging.warning(
                    f"Labels {missing_labels} found in RAP volume but not in "
                    f"methods config. Base params will be used for these labels."
                )

    def rap_multistep_propagate(self, line, prev_direction):
        """
        Propagate within the RAP region using modified parameters.

        Parameters
        ----------
        line : list
            The current streamline.
        prev_direction : np.ndarray
            The previous tracking direction.

        Returns
        -------
        line : list
            The extended streamline.
        prev_direction : np.ndarray
            The last direction.
        is_line_valid : bool
            Whether the line is valid.
        """
        # Switch to RAP parameters
        label = self._get_label(line[-1],
                                self.propagator.space,
                                self.propagator.origin)
        if label <= 0:
            return line, prev_direction, False

        # Logging debug when label changes        # Apply the parameters of the RAP labels
        if label != self._current_label:
            if self._current_label is not None:
                logging.debug(f"STEP[{self._total_steps}] label={self._current_label}"
                              f", algo={self.propagator.algo}"
                              f", theta (rad)={self.propagator.theta}"
                              f", vox step size={self.propagator.step_size}"
                              f" -> switching label to label {label}")
            self._current_label = label

        # Switch propagator based on label
        if str(label) in self._propagators:
            new_propagator = self._propagators[str(label)]
            if new_propagator is not self.propagator:
                new_propagator.line_rng_generator = self.propagator.line_rng_generator
                self.propagator = new_propagator
                logging.debug(f"RAP propagator switched to label {label}")
        else:
            new_propagator = self._propagators[self._propagators.keys()[0]]
            if new_propagator is not self.propagator:
                new_propagator.line_rng_generator = self.propagator.line_rng_generator
                self.propagator = new_propagator
                logging.debug(f"RAP propagator switched to default label {self._propagators.keys()[0]}")

        # Normalize previous direction representation when switching
        # propagator families.
        #
        # ODF propagators rely on TrackingDirection.index to lookup angular
        # neighborhoods on the discrete sphere (tracking_neighbours). Tensor
        # propagators only need the Cartesian direction and can operate on a
        # plain ndarray.
        #
        # Therefore:
        # - Tensor -> ODF: wrap/quantize direction using prepare_backward so
        #   an index is available on the target ODF sphere.
        # - ODF -> Tensor: drop the index and keep only Cartesian components
        #   to avoid carrying stale sphere metadata across models.
        prev_direction_has_index = getattr(prev_direction, 'index', None) is not None
        if hasattr(self.propagator, 'tracking_neighbours') and not prev_direction_has_index:
            prev_direction = self.propagator.prepare_backward(line, prev_direction)
        elif not hasattr(self.propagator, 'tracking_neighbours') and prev_direction_has_index:
            prev_direction = np.asarray(prev_direction)

        # Perform propagation with new parameters
        new_pos, new_dir, is_direction_valid = self.propagator.propagate(
            line, prev_direction)

        # Add the new point to the line
        if is_direction_valid:
            line.append(new_pos)
            self._total_steps += 1
            return line, new_dir, True
        return line, prev_direction, False

    def _get_label(self, curr_pos, space, origin):
        """
        Receive label (int) at current position in RAP label volume.

        Parameters
        ----------
        curr_pos: np.ndarray
            This is the current 3D position of the streamline.

        space: Space
            Coordinate space (here Space.VOX.).

        origin: Origin
            Origin convention ('center').

        Returns
        -------
        int
            The integer label at current position.
        """
        v = self.rap_volume.get_value_at_coordinate(
            *curr_pos, space=space, origin=origin)
        try:
            return int(v)
        except Exception:
            return int(np.round(v))


class RAPGraph(RAP):
    def __init__(self, mask_rap, propagator, max_nbr_pts, neighboorhood_size):
        super().__init__(mask_rap, propagator, max_nbr_pts)
        self.neighboorhood_size = neighboorhood_size

    def rap_multistep_propagate(self, line, prev_direction):
        raise NotImplementedError
