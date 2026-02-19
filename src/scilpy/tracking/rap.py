# -*- coding: utf-8 -*-

import json
import logging
import numpy as np
from copy import deepcopy
from dipy.core.geometry import math
from scilpy.tracking.propagator import get_sphere_neighbours


class RAP:
    def __init__(self, mask_rap, propagator, max_nbr_pts):
        """
        RAP_mask: DataVolume
            HRegion-Adaptive Propagation tractography volume.
        """
        self.rap_mask = mask_rap
        self.propagator = propagator
        self.max_nbr_pts = max_nbr_pts

    def is_in_rap_region(self, curr_pos, space, origin):
        return self.rap_mask.get_value_at_coordinate(
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
    def __init__(self, mask_rap, propagator, max_nbr_pts, step_size):
        """
        Step size: float
            The step size inside the RAP mask. Could be different from the step
            size elsewhere. In voxel world.
        """
        super().__init__(mask_rap, propagator, max_nbr_pts)
        self.step_size = step_size

    def rap_multistep_propagate(self, line, prev_direction):
        is_line_valid = True
        if len(line) > 3:
            pos = line[-2] + self.step_size * np.array(prev_direction)
            line[-1] = pos
            return line, prev_direction, is_line_valid
        return line, prev_direction, is_line_valid


class RAPSwitch(RAP):
    """RAP class that switches tracking parameters when inside the RAP mask."""
    def __init__(self, mask_rap, propagator, max_nbr_pts, rap_params_file, rap_labels=None):
        """
        Parameters
        ----------
        mask_rap : DataVolume
            Region-Adaptive Propagation mask.
        propagator : Propagator
            The propagator used for tracking.
        max_nbr_pts : int
            Maximum number of points per streamline.
        rap_params_file : str
            Path to JSON file containing RAP parameters.
            Expected format: {
                "step_size": float,
                "theta": float (in degrees)
            }
        """
        super().__init__(mask_rap, propagator, max_nbr_pts)

        # Load parameters from JSON file
        with open(rap_params_file, 'r') as f:
            rap_params = json.load(f)
        cfg = rap_params

        # Store original parameters
        self.original_step_size = propagator.step_size
        self.original_theta = propagator.theta

        # Store RAP parameters (convert step size to voxel space if needed)
        self.rap_step_size = rap_params.get('step_size', self.original_step_size)
        # Convert theta from degrees to radians
        self.rap_theta = math.radians(rap_params.get('theta',
                                                     math.degrees(self.original_theta)))
        
        self.rap_mask = mask_rap
        self.rap_labels = rap_labels
        self._mode = 'label' if rap_labels is not None else 'mask'

        self._base = {
            'step_size': propagator.step_size,
            'theta': propagator.theta,
            'algo' : getattr(propagator, 'algo', None),
            'tracking_neighbours' : getattr(propagator, 'tracking_neighbours', None)
        }

        if self._mode == 'label':
            self.default_cfg = cfg.get('default', {})
            self.methods_cfg = cfg.get('methods', {})
        else:
            self.rap_cfg = cfg

        logging.info("RAP parameters loaded:")
        logging.info(f"  Original step_size: {self.original_step_size:.3f}, "
                     f"RAP step_size: {self.rap_step_size:.3f}")
        logging.info(f"  Original theta: {math.degrees(self.original_theta):.2f}°, "
                     f"RAP theta: {math.degrees(self.rap_theta):.2f}°")

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
        is_line_valid = True

        # We allow RAP to extend the streamline while it stays inside the RAP region
        # In mask mode: "inside" means rap_mask > 0
        # In label mode: "inside" means label > 0 and we can switch config per label
        while len(line) < self.max_nbr_pts:
            curr_pos = line[-1]

            # Select config depending on RAP mode
            if self._mode == 'label':
                label = self._get_label(curr_pos, self.propagator.space, self.propagator.origin)
                if label <= 0:
                    break
                cfg = self._merge_cfg(label)
            else:
                # Classic binary RAP mask behaviour
                if not self.is_in_rap_region(curr_pos, self.propagator.space, self.propagator.origin):
                    break
                cfg = self.rap_cfg

            # Apply selected params for ONE step, then restore
            self._apply_cfg(cfg)
            try:
                new_pos, new_dir, valid = self.propagator.propagate(line, prev_direction)
            finally:
                self._restore_base()

            is_line_valid = is_line_valid and valid
            if not valid:
                break

            line.append(new_pos)
            prev_direction = new_dir

        return line, prev_direction, is_line_valid
        
    def _get_label(self, curr_pos, space, origin):
        v = self.rap_labels.get_value_at_coordinate(*curr_pos, space=space, origin=origin)
        try:
            return int(v)
        except Exception:
            return int(np.round(v))
        
    def _merge_cfg(self, label):
        cfg = deepcopy(self.default_cfg)
        override = self.methods_cfg.get(str(label), {})
        cfg.update(override)
        return cfg
    
    def _apply_cfg(self, cfg):
        if 'step_size' in cfg and cfg['step_size'] is not None:
            self.propagator.step_size = float(cfg['step_size'])
        if 'algo' in cfg and cfg['algo'] is not None:
            self.propagator.algo = str(cfg['algo'])
        if 'theta' in cfg and cfg['theta'] is not None:
            theta_rad = np.deg2rad(float(cfg['theta']))
            self.propagator.theta = theta_rad
            # theta change => neighbours change
            self.propagator.tracking_neighbours = get_sphere_neighbours(self.propagator.sphere, self.propagator.theta)

    def _restore_base(self):
        self.propagator.step_size = self._base['step_size']
        self.propagator.theta = self._base['theta']
        if self._base['algo'] is not None:
            self.propagator.algo = self._base['algo']
        if self._base['tracking_neighbours'] is not None:
            self.propagator.tracking_neighbours = self._base['tracking_neighbours']

    def is_in_rap_region(self, curr_pos, space, origin):
        """Override base class to support label-mode when rap_mask is None.
        Tracker uses this to decide whether to enter/exit RAP.
        - mask mode: inside if rap_mask > 0
        - label mode: inside if rap_labels label > 0
        """
        if self._mode == 'label':
            if self.rap_labels is None:
                return False
            val = self.rap_labels.get_value_at_coordinate(
                *curr_pos, space=space, origin=origin)
            return val > 0

        # mask mode (legacy)
        if self.rap_mask is None:
            return False
        return self.rap_mask.get_value_at_coordinate(
            *curr_pos, space=space, origin=origin) > 0

class RAPGraph(RAP):
    def __init__(self, mask_rap, propagator, max_nbr_pts, neighboorhood_size):
        super().__init__(mask_rap, propagator, max_nbr_pts)
        self.neighboorhood_size = neighboorhood_size

    def rap_multistep_propagate(self, line, prev_direction):
        raise NotImplementedError
