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

    def get_label_at(self, curr_pos, space, origin):
        """Return integer label at position. 0 means outside RAP"""
        val = self.rap_mask.get_value_at_coordinate(
            *curr_pos, space=space, origin=origin)
        try:
            return int(val)
        except Exception:
            return 0

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
    """RAP class that switches tracking parameters when inside the RAP mask, using a label-map
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
            
    RAP mask convention (label-map):
        0: outside RAP
        1..N: region labels

    JSON formats supported:
    (A) Legacy single policy (applied for any label > 0):
        {"step_size": 0.5, "theta": 35, "algo": "prob"}
    (B) Multi-label policies:
        {
          "1": {"step_size": 0.5, "theta": 35, "algo": "det"},
          "2": {"step_size": 0.5, "theta": 35, "algo": "prob"},
          "3": {"step_size": 0.25, "theta": 10, "algo": "prob"}
        }

    `theta` is specified in degrees in JSON and converted to radians.
    """

    def __init__(self, mask_rap, propagator, max_nbr_pts, rap_params_file):
        super().__init__(mask_rap, propagator, max_nbr_pts)

        if rap_params_file is None:
            raise ValueError('RAPSwitch requires rap_params_file.')

        with open(rap_params_file, 'r') as f:
            raw = json.load(f)

        # Normalized policies: {0: legacy_policy} or {label:int -> policy:dict}
        self._policies = self._parse_policies(raw)

    def _parse_policies(self, raw):
        if not isinstance(raw, dict) or len(raw) == 0:
            raise ValueError('Invalid RAP JSON: expected a non-empty object (dict).')

        # (A) Legacy single policy for any label > 0
        if 'step_size' in raw and 'theta' in raw:
            pol = {
                'step_size': float(raw['step_size']),
                'theta': float(raw['theta']),  # degrees
            }
            if 'algo' in raw and raw['algo'] is not None:
                algo = str(raw['algo']).lower()
                if algo not in ['det', 'prob']:
                    raise ValueError("Legacy RAPSwitch 'algo' must be 'det' or 'prob'.")
                pol['algo'] = algo
            return {0: pol}

        # (B) Multi-label policies
        policies = {}
        for k, v in raw.items():
            label = int(k)
            if label <= 0:
                continue
            if not isinstance(v, dict):
                raise ValueError(f'RAPSwitch policy for label {label} must be an object (dict).')
            if 'step_size' not in v or 'theta' not in v:
                raise ValueError(
                    f"RAPSwitch policy for label {label} must contain 'step_size' and 'theta' (deg).")

            pol = dict(v)
            pol['step_size'] = float(v['step_size'])
            pol['theta'] = float(v['theta'])  # degrees

            if 'algo' in v and v['algo'] is not None:
                algo = str(v['algo']).lower()
                if algo not in ['det', 'prob']:
                    raise ValueError(
                        f"RAPSwitch policy for label {label}: 'algo' must be 'det' or 'prob'.")
                pol['algo'] = algo

            policies[label] = pol

        if len(policies) == 0:
            raise ValueError('Invalid RAP JSON: no valid label policies found.')

        return policies

    def _get_label(self, curr_pos, space, origin):
        # DataVolume may return float; enforce integer labels.
        val = self.rap_mask.get_value_at_coordinate(
            *curr_pos, space=space, origin=origin)
        try:
            return int(val)
        except Exception:
            return 0

    def _select_policy(self, label):
        if label <= 0:
            return None
        # Legacy mode: apply same policy to any label > 0
        if 0 in self._policies:
            return self._policies[0]
        return self._policies.get(label, None)

    def _snapshot_propagator_params(self):
        """Best-effort snapshot so we can restore after RAP."""
        snap = {}
        for key in ['algo', 'theta', 'step_size']:
            if hasattr(self.propagator, key):
                snap[key] = getattr(self.propagator, key)
        # Also snapshot tracking_neighbours if present (derived from theta).
        if hasattr(self.propagator, 'tracking_neighbours'):
            snap['tracking_neighbours'] = getattr(self.propagator,
                                                 'tracking_neighbours')
        return snap

    def _restore_propagator_params(self, snap):
        for k, v in snap.items():
            setattr(self.propagator, k, v)

    def _apply_policy(self, policy):
        """Apply policy keys directly to propagator.

        Special cases:
        - theta: degrees -> radians + recompute tracking_neighbours.
        """
        theta_changed = False

        # Apply algo first (if present)
        if 'algo' in policy and hasattr(self.propagator, 'algo'):
            self.propagator.algo = policy['algo']

        # Apply step_size (if present)
        if 'step_size' in policy and hasattr(self.propagator, 'step_size'):
            self.propagator.step_size = float(policy['step_size'])

        # Apply theta (degrees -> radians)
        if 'theta' in policy and hasattr(self.propagator, 'theta'):
            self.propagator.theta = math.radians(float(policy['theta']))
            theta_changed = True

        # Apply any other custom keys if propagator has the attribute.
        for k, v in policy.items():
            if k in ['algo', 'step_size', 'theta']:
                continue
            if hasattr(self.propagator, k):
                setattr(self.propagator, k, v)

        # If theta changed, recompute tracking_neighbours (Arnaud's approach).
        if theta_changed:
            from scilpy.tracking.propagator import get_sphere_neighbours
            if hasattr(self.propagator, 'sphere'):
                self.propagator.tracking_neighbours = get_sphere_neighbours(
                    self.propagator.sphere, self.propagator.theta)

    def rap_multistep_propagate(self, line, prev_direction):
        """Propagate inside RAP using the policy corresponding to the current label.

        Uses Arnaud's propagate API:
            new_pos, new_dir, is_direction_valid = propagator.propagate(line, prev_direction)
        """
        is_line_valid = True

        space = getattr(self.propagator, 'space', None)
        origin = getattr(self.propagator, 'origin', None)

        # Safety: need at least one point
        if len(line) == 0:
            return line, prev_direction, False

        # Save propagator params once for the whole RAP segment
        snap = self._snapshot_propagator_params()

        for _ in range(self.max_nbr_pts):
            curr_pos = line[-1]
            label = self._get_label(curr_pos, space, origin)
            if label <= 0:
                break

            policy = self._select_policy(label)
            if policy is None:
                break

            self._apply_policy(policy)

            new_pos, new_dir, is_direction_valid = self.propagator.propagate(
                line, prev_direction)
            if not is_direction_valid:
                is_line_valid = False
                break

            line.append(new_pos)
            prev_direction = new_dir

        # Restore original propagator params after leaving RAP
        self._restore_propagator_params(snap)

        return line, prev_direction, is_line_valid


class RAPGraph(RAP):
    def __init__(self, mask_rap, propagator, max_nbr_pts, neighboorhood_size):
        super().__init__(mask_rap, propagator, max_nbr_pts)
        self.neighboorhood_size = neighboorhood_size

    def rap_multistep_propagate(self, line, prev_direction):
        raise NotImplementedError
