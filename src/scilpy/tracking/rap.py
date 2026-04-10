# -*- coding: utf-8 -*-

import logging
import numpy as np
from copy import deepcopy
from scilpy.tracking.propagator import get_sphere_neighbours


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
        self._current_cfg = {}

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
        rap_params : dict
            Dictionary containing RAP parameters, loaded from
            the JSON policies file.
            Expected format:
            {
                "methods": {
                "1": {"propagator": "ODF", "filename": str, "sh_basis": str,
                        "algo": str, "theta": float, "step_size": float},
                "2": {"propagator": "ODF", "filename": str, "sh_basis": str,
                        "algo": str, "theta": float, "step_size": float},
                ...
                }
            }
            If 'propagator' is 'ODF', the fODF file specified
            in 'filename' is used.
            'sh_basis' defaults to 'descoteaux07_legacy'.
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
                              if label not in self._propagators.keys()]
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
        # Apply the parameters of the RAP labels
        cfg = self._get_label_cfg(label)

        # Logging debug when label changes
        if label != self._current_label:
            if self._current_label is not None:
                logging.debug(f"STEP[{self._total_steps}] label={self._current_label}"
                              f", algo={self._current_cfg.get('algo')}"
                              f", Theta (rad)={self._current_cfg.get('theta')}"
                              f", vox step size={self._current_cfg.get('step_size')}"
                              f" -> switching label to label {label}")
            self._current_label = label
            self._current_cfg = cfg

        # Switch propagator based on label
        if str(label) in self._propagators:
            new_propagator = self._propagators[str(label)]
            if new_propagator is not self.propagator:
                new_propagator.line_rng_generator = self.propagator.line_rng_generator
                self.propagator = new_propagator
                self.propagator.tracking_neighbours = get_sphere_neighbours(
                    self.propagator.sphere, self.propagator.theta)
                logging.debug(f"RAP propagator switched to label {label}")

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

    def _get_label_cfg(self, label):
        """
        Get the configuration for the given label from the JSON policy.

        Parameters
        ----------
        label: int
            Integer of label at current position.

        Returns
        -------
        dict
            Configuration dict for the given label from the JSON policy,
            with keys 'algo', 'theta', 'step_size'.
        """
        override = {
                'step_size': self._propagators[str(label)].step_size,
                'algo': self._propagators[str(label)].algo,
                'theta': self._propagators[str(label)].theta
            }

        if override is None:
            return {
                'step_size': self._base['step_size'],
                'algo': self._base['algo'],
                'theta': float(np.degrees(self._base['theta']))
            }
        return deepcopy(override)


class RAPGraph(RAP):
    def __init__(self, mask_rap, propagator, max_nbr_pts, neighboorhood_size):
        super().__init__(mask_rap, propagator, max_nbr_pts)
        self.neighboorhood_size = neighboorhood_size

    def rap_multistep_propagate(self, line, prev_direction):
        raise NotImplementedError
