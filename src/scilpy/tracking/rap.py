# -*- coding: utf-8 -*-

import json
import logging
import numpy as np
from dipy.core.geometry import math


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
    def __init__(self, mask_rap, propagator, max_nbr_pts, rap_params_file):
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
        """
        super().__init__(mask_rap, propagator, max_nbr_pts)

        # RAP policies can be provided in two formats:
        # (1) Legacy: single policy applied to label above zero
            # {"step_size": float, "theta": float (deg), "algo": "det"|"prob" (optional)}
        # (2) Multi-label policies: label map
        #     {"1": {"step_size": .., "theta": .., "algo": ..},
        #      "2": {...}, ... }
        with open(rap_params_file, 'r') as f:
            raw = json.load(f)

        self._policies = self._parse_policies(raw)
        self._last_label = None
        self._prev_params = None  # (algo, theta_rad, step_size)

        # We want to keep a small log so we know which mode is used
        if 0 in self._policies:
            logging.info("RAPSwitch loaded legacy single-policy JSON.")
        else:
            logging.info("RAPSwitch loaded multi-label policies for labels: %s",
                         sorted(self._policies.keys()))

    def _parse_policies(self, raw):
        # Legacy single-policy format
        if isinstance(raw, dict) and 'step_size' in raw and 'theta' in raw:
            algo = raw.get('algo', None)
            if algo is not None and algo not in ['det', 'prob']:
                raise ValueError("RAPSwitch legacy policy 'algo' must be 'det' or 'prob'")
            step_size = float(raw['step_size'])
            theta_rad = math.radians(float(raw['theta']))
            return {0: {'algo': algo, 'step_size': step_size, 'theta_rad': theta_rad}}
        
            # Multi-label format
        if not isinstance(raw, dict):
            raise ValueError('RAPSwitch policies JSON must be a dict')

        policies = {}
        for k, v in raw.items():
            label = int(k)
            if not isinstance(v, dict):
                raise ValueError(f'RAPSwitch policy for label {label} must be a dict')
            algo = v.get('algo', None)
            if algo is not None and algo not in ['det', 'prob']:
                raise ValueError(f"RAPSwitch policy for label {label}: 'algo' must be 'det' or 'prob'")
            if 'step_size' not in v or 'theta' not in v:
                raise ValueError(f"RAPSwitch policy for label {label} must contain 'step_size' and 'theta' (deg).")
            step_size = float(v['step_size'])
            theta_rad = math.radians(float(v['theta']))
            policies[label] = {'algo': algo, 'step_size': step_size, 'theta_rad': theta_rad}

        return policies

    def _get_label(self, curr_pos, space, origin):
        # we give the option for rap_mask to be a binary mask or an integer label map
        return int(self.rap_mask.get_value_at_coordinate(
            *curr_pos, space=space, origin=origin))

    def _save_prev_params_if_needed(self, label):
        if self._last_label == label:
            return
        self._prev_params = (
            getattr(self.propagator, 'algo', None),
            getattr(self.propagator, 'theta', None),
            getattr(self.propagator, 'step_size', None)
        )
        self._last_label = label

    def _apply_policy(self, policy):
        # Apply algo
        if policy.get('algo', None) is not None:
            if hasattr(self.propagator, 'set_algo'):
                self.propagator.set_algo(policy['algo'])
            else:
                self.propagator.algo = policy['algo']

        # Apply theta
        if hasattr(self.propagator, 'set_theta'):
            self.propagator.set_theta(policy['theta_rad'])
        else:
            self.propagator.theta = policy['theta_rad']

        # Apply step size
        if hasattr(self.propagator, 'set_step_size'):
            self.propagator.set_step_size(policy['step_size'])
        else:
            self.propagator.step_size = policy['step_size']

    def _restore_prev_params(self):
        if self._prev_params is None:
            self._last_label = None
            return
        algo, theta, step = self._prev_params

        if algo is not None:
            if hasattr(self.propagator, 'set_algo'):
                self.propagator.set_algo(algo)
            else:
                self.propagator.algo = algo

        if theta is not None:
            if hasattr(self.propagator, 'set_theta'):
                self.propagator.set_theta(theta)
            else:
                self.propagator.theta = theta

        if step is not None:
            if hasattr(self.propagator, 'set_step_size'):
                self.propagator.set_step_size(step)
            else:
                self.propagator.step_size = step

        self._prev_params = None
        self._last_label = None
   
    def rap_multistep_propagate(self, line, prev_direction):
        is_line_valid = True
        
        # Apply RAP policies while we remain in RAP region
        for _ in range(self.max_nbr_pts):
            curr_pos = line[-1]
            label = self._get_label(curr_pos, self.propagator.space, 
                                    self.propagator.origin)
            if label <= 0:
                break

            # Choose policy
            if 0 in self._policies:
                # Legacy single policy mode
                policy = self._policies[0]
            else:
                policy = self._policies.get(label, None)
                if policy is None:
                    # If no policy for this label then we stop RAP here
                    break

            self._save_prev_params_if_needed(label)
            self._apply_policy(policy)

            new_pos, new_dir, is_direction_valid = \
                self.propagator.propagate(line, prev_direction)
       
            if not is_direction_valid:
                is_line_valid = False
                break
        
            line.append(new_pos)
            prev_direction = new_dir

        # Restore original parameters after leaving RAP
        self._restore_prev_params()
        return line, prev_direction, is_line_valid


class RAPGraph(RAP):
    def __init__(self, mask_rap, propagator, max_nbr_pts, neighboorhood_size):
        super().__init__(mask_rap, propagator, max_nbr_pts)
        self.neighboorhood_size = neighboorhood_size

    def rap_multistep_propagate(self, line, prev_direction):
        raise NotImplementedError
