# -*- coding: utf-8 -*-
import logging

from dipy.io.utils import create_tractogram_header, get_reference_info
from dipy.tracking import utils as track_utils
from dipy.tracking.streamlinespeed import compress_streamlines, length
import nibabel as nib
from nibabel.streamlines.tractogram import LazyTractogram
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist)


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


def verify_tracking_args(parser, args):
    assert_inputs_exist(parser, [args.in_sh, args.in_seed, args.in_mask])
    assert_outputs_exist(parser, args, args.out_tractogram)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    if not args.min_length > 0:
        parser.error('minL must be > 0, {}mm was provided.'
                     .format(args.min_length))
    if args.max_length < args.min_length:
        parser.error('maxL must be > than minL, (minL={}mm, maxL={}mm).'
                     .format(args.min_length, args.max_length))

    if args.compress:
        if args.compress < 0.001 or args.compress > 1:
            logging.warning(
                'You are using an error rate of {}.\nWe recommend setting it '
                'between 0.001 and 1.\n0.001 will do almost nothing to the '
                'tracts while 1 will higly compress/linearize the tracts'
                .format(args.compress))

    if args.npv and args.npv <= 0:
        parser.error('Number of seeds per voxel must be > 0.')

    if args.nt and args.nt <= 0:
        parser.error('Total number of seeds must be > 0.')


def load_mask_and_verify_anisotropy(in_mask, in_sh):
    mask_img = nib.load(in_mask)
    mask_data = get_data_as_mask(mask_img, dtype=bool)

    # Make sure the mask is isotropic. Else, the strategy used
    # when providing information to dipy (i.e. working as if in voxel space)
    # will not yield correct results.
    fodf_sh_img = nib.load(in_sh)
    if not np.allclose(np.mean(fodf_sh_img.header.get_zooms()[:3]),
                       fodf_sh_img.header.get_zooms()[0], atol=1e-03):
        raise ValueError(
            'SH file is not isotropic. Tracking cannot be ran robustly.')

    voxel_size = fodf_sh_img.header.get_zooms()[0]

    return mask_data, voxel_size


def prepare_seeds(in_seed, random_seed, npv=None, nt=None):
    if npv:
        nb_seeds = npv
        seed_per_vox = True
    elif nt:
        nb_seeds = nt
        seed_per_vox = False
    else:
        nb_seeds = 1
        seed_per_vox = True

    seed_img = nib.load(in_seed)
    seeds = track_utils.random_seeds_from_mask(
        seed_img.get_fdata(dtype=np.float32),
        np.eye(4),
        seeds_count=nb_seeds,
        seed_count_per_voxel=seed_per_vox,
        random_seed=random_seed)

    return seed_img, seeds


def save_results(streamlines, voxel_size, ref, min_length, max_length,
                 save_seeds, compress, out_tractogram):
    scaled_min_length = min_length / voxel_size
    scaled_max_length = max_length / voxel_size

    if save_seeds:
        filtered_streamlines, seeds = \
            zip(*((s, p) for s, p in streamlines
                  if scaled_min_length <= length(s) <= scaled_max_length))
        data_per_streamlines = {'seeds': lambda: seeds}
    else:
        filtered_streamlines = \
            (s for s in streamlines
             if scaled_min_length <= length(s) <= scaled_max_length)
        data_per_streamlines = {}

    if compress:
        filtered_streamlines = (compress_streamlines(s, compress)
                                for s in filtered_streamlines)

    tractogram = LazyTractogram(lambda: filtered_streamlines,
                                data_per_streamlines,
                                affine_to_rasmm=ref.affine)

    filetype = nib.streamlines.detect_format(out_tractogram)
    reference = get_reference_info(ref)
    header = create_tractogram_header(filetype, *reference)

    # Use generator to save the streamlines on-the-fly
    nib.streamlines.save(tractogram, out_tractogram, header=header)
