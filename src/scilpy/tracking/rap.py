# -*- coding: utf-8 -*-

import numpy as np
from scilpy.tracking.utils import TrackingDirection



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
        if len(line)>3:
            pos = line[-2] + self.step_size * np.array(prev_direction)
            line[-1] = pos
            return line, prev_direction, is_line_valid
        return line, prev_direction, is_line_valid


class RAPGraph(RAP):
    def __init__(self, mask_rap, rap_img, propagator, max_nbr_pts, fodf, reps, alpha):
        """
        RAPGraph class for the quantum Graph solution for a region.

        Parameters
        ----------
        fodf: DataVolume
            The FODF volume used to compute the RAP.
        reps: int
            Number of repetitions used in the quantum circuit.
        alpha: float
            Initial paramater to search the cost landscape.
        """
        super().__init__(mask_rap, propagator, max_nbr_pts)
       

        self.fodf = fodf
        self.rap_img = rap_img
        self.reps = reps
        self.alpha = alpha


    def rap_multistep_propagate(self, line, prev_direction):
        try:
            from quactography.solver.rap_tracking import quack_rap
        except ImportError:
            raise ImportError("quactography is not installed. "
                              "Please install it to use RAPGraph.\n"
                              "Add: Follow instructions here: https://github.com/scilus/quactography")
        
        prev_dir = np.array(prev_direction)
        seg, prev_dir, is_line_valid = quack_rap(self.rap_img, self.fodf, line[-1].round().astype(int),
                                                  reps = self.reps,
                                                  alpha = self.alpha, 
                                                  prev_direction = prev_dir,
                                                  theta = self.propagator.theta,
                                                  threshold = self.propagator.sf_threshold)
        line.extend(seg)
        last_dir = TrackingDirection(prev_direction)
        
        return line, last_dir, is_line_valid
