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
            Expected format: {
                "step_size": float,
                "theta": float (in degrees)
            }
        """
        super().__init__(mask_rap, propagator, max_nbr_pts)

        # Load parameters from JSON file
        with open(rap_params_file, 'r') as f:
            rap_params = json.load(f)

        # Store original parameters
        self.original_step_size = propagator.step_size
        self.original_theta = propagator.theta

        # Store RAP parameters (convert step size to voxel space if needed)
        self.rap_step_size = rap_params.get('step_size', self.original_step_size)
        # Convert theta from degrees to radians
        self.rap_theta = math.radians(rap_params.get('theta',
                                                     math.degrees(self.original_theta)))

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
        self.propagator.step_size = self.rap_step_size
        self.propagator.theta = self.rap_theta

        # Update tracking neighbours with new theta
        from scilpy.tracking.propagator import get_sphere_neighbours
        self.propagator.tracking_neighbours = get_sphere_neighbours(
            self.propagator.sphere, self.rap_theta)

        # Perform propagation with new parameters
        new_pos, new_dir, is_direction_valid = \
            self.propagator.propagate(line, prev_direction)

        # Restore original parameters
        self.propagator.step_size = self.original_step_size
        self.propagator.theta = self.original_theta
        self.propagator.tracking_neighbours = get_sphere_neighbours(
            self.propagator.sphere, self.original_theta)

        # Add the new point to the line
        if is_direction_valid:
            line.append(new_pos)
            return line, new_dir, True
        else:
            return line, prev_direction, False


class RAPGraph(RAP):
    def __init__(self, mask_rap, propagator, max_nbr_pts, neighboorhood_size):
        super().__init__(mask_rap, propagator, max_nbr_pts)
        self.neighboorhood_size = neighboorhood_size

    def rap_multistep_propagate(self, line, prev_direction):
        raise NotImplementedError
