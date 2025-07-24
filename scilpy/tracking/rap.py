# -*- coding: utf-8 -*-

import numpy as np


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
    def __init__(self, mask_rap, propagator, max_nbr_pts, neighboorhood_size, fodf):
        super().__init__(mask_rap, propagator, max_nbr_pts)
        from quactography.scripts.quac_matrix_adj_build import build_mat

        self.neighboorhood_size = neighboorhood_size
        self.fodf = fodf


    def rap_multistep_propagate(self, line, prev_direction):
        is_line_valid = True
        list_of_end_points = [] #from end point mask, end of the ROI to create the graph with anatomically possible line
    
        if self.max_nbr_pts + len(list_of_end_points)> 17: 
            print("RAPGraph: max number of points exceeded")
            is_line_valid = False
            return line, prev_direction, is_line_valid
        line.extend(quack_rap(self.mask_rap, self.fodf, line[-1], list_of_end_points))
        return line, prev_direction, is_line_valid
