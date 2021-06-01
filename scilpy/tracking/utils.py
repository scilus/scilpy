# -*- coding: utf-8 -*-


class TrackingDirection(list):

    """
    Tracking direction use as 3D cartesian direction (list(x,y,z))
    and has an index to work with discrete sphere.
    """

    def __init__(self, cartesian, index=None):
        super(TrackingDirection, self).__init__(cartesian)
        self.index = index


class TrackingParams(object):
    """
    Container for tracking parameters.
    """
    def __init__(self):
        self.random = None
        self.skip = None
        self.algo = None
        self.mask_interp = None
        self.field_interp = None
        self.theta = None
        self.sf_threshold = None
        self.sf_threshold_init = None
        self.step_size = None
        self.rk_order = None
        self.max_length = None
        self.min_length = None
        self.max_nbr_pts = None
        self.min_nbr_pts = None
        self.is_single_direction = None
        self.nbr_seeds = None
        self.nbr_seeds_voxel = None
        self.nbr_streamlines = None
        self.max_no_dir = None
        self.is_all = None
        self.is_keep_single_pts = None
        self.mmap_mode = None
