# -*- coding: utf-8 -*-

import json
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
    """RAP class that switches tracking parameters when inside the RAP mask or RAP label."""
    def __init__(self, rap_volume, propagator, max_nbr_pts, rap_params_file):
        """
        Parameters
        ----------
        rap_volume : DataVolume
            Region-Adaptive Propagation mask.
        propagator : Propagator
            The propagator used for tracking.
        max_nbr_pts : int
            Maximum number of points per streamline.
        rap_params_file : str
            Path to JSON file containing RAP parameters.
            "methods" is optionnal, if not provided, "default" will be applied
            Expected format:
            {
                "methods": {
                  "1": {"algo": str, "theta": float, "step_size": float},
                  "2": {"algo": str, "theta": float, "step_size": float},
                  ...
                }
            }
        """
        super().__init__(rap_volume, propagator, max_nbr_pts)

        # Load parameters from JSON file
        with open(rap_params_file, 'r') as f:
            rap_params = json.load(f)

        self._base = {
            'step_size': propagator.step_size,
            'theta': propagator.theta,
            'algo': getattr(propagator, 'algo', None),
            'tracking_neighbours': getattr(propagator, 'tracking_neighbours', None)
        }
        self.methods_cfg = rap_params.get('methods', {})
        logging.info("RAP parameters loaded:")

        # Check if all labels in the volume are covered by the configuration
        unique_labels = np.unique(rap_volume.data)
        # Remove 0 (background) and convert to int
        unique_labels = [int(label) for label in unique_labels if label > 0]

        if unique_labels:
            missing_labels = [label for label in unique_labels
                              if str(label) not in self.methods_cfg]
            if missing_labels:
                logging.warning(
                    f"Labels {missing_labels} found in RAP volume but not in "
                    f"methods config. Base parameters will be used for these labels."
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
        label = self._get_label(line[-1], self.propagator.space, self.propagator.origin)
        if label <= 0:
            return line, prev_direction, False
        # Apply the parameters of the RAP labels
        cfg = self._merge_cfg(label)

        # Perform propagation with new parameters
        self._apply_cfg(cfg)
        new_pos, new_dir, is_direction_valid = self.propagator.propagate(line, prev_direction)

        # Add the new point to the line
        if is_direction_valid:
            line.append(new_pos)
            if label != self._current_label:
                if self._current_label is not None:
                    logging.debug(f"STEP[{self._total_steps}] label={self._current_label} algo={self._current_cfg.get('algo')} theta={self._current_cfg.get('theta')} step={self._current_cfg.get('step_size')}")
                self._current_label = label
                self._current_cfg = cfg
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
        v = self.rap_volume.get_value_at_coordinate(*curr_pos, space=space, origin=origin)
        try:
            return int(v)
        except Exception:
            return int(np.round(v))

    def _merge_cfg(self, label):
        """
        Merge the default configuration with the label-specific cfg override from the JSON policy.

        Parameters
        ----------
        label: int
            Integer of label at current position.

        Returns
        -------
        dict
            Configuration dict with keys 'algo', 'theta', 'step_size'.
        """
        override = self.methods_cfg.get(str(label))
        if override is None:
            return {
                'step_size': self._base['step_size'],
                'algo': self._base['algo'],
                'theta': float(np.degrees(self._base['theta']))
            }
        return deepcopy(override)

    def _apply_cfg(self, cfg):
        """
        Temporarily apply a label configuration to the propagator.

        Parameters
        ----------
        cfg: dict
            Configuration dict with keys 'algo', 'theta', 'step_size'.
        """
        if 'step_size' in cfg and cfg['step_size'] is not None:
            self.propagator.step_size = float(cfg['step_size'])
        if 'algo' in cfg and cfg['algo'] is not None:
            self.propagator.algo = str(cfg['algo'])
        if 'theta' in cfg and cfg['theta'] is not None:
            theta_rad = np.deg2rad(float(cfg['theta']))
            self.propagator.theta = theta_rad
            # theta change => neighbours change
            self.propagator.tracking_neighbours = get_sphere_neighbours(self.propagator.sphere, self.propagator.theta)


class RAPGraph(RAP):
    def __init__(self, mask_rap, propagator, max_nbr_pts, neighboorhood_size):
        super().__init__(mask_rap, propagator, max_nbr_pts)
        self.neighboorhood_size = neighboorhood_size

    def rap_multistep_propagate(self, line, prev_direction):
        raise NotImplementedError
