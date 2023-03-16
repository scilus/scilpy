#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local streamline HARDI tractography.
The tracking direction is chosen in the aperture cone defined by the
previous tracking direction and the angular constraint.

Algo 'eudx': the peak from the spherical function (SF) most closely aligned
to the previous direction.
Algo 'det': the maxima of the spherical function (SF) the most closely aligned
to the previous direction.
Algo 'prob': a direction drawn from the empirical distribution function defined
from the SF.

NOTE: eudx can be used with pre-computed peaks from fodf as well as
evecs_v1.nii.gz from scil_compute_dti_metrics.py (experimental).

All the input nifti files must be in isotropic resolution.
"""

import argparse
import logging

from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.utils import (get_reference_info,
                           create_tractogram_header)
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamlinespeed import length, compress_streamlines
from dipy.tracking import utils as track_utils
import nibabel as nib
from nibabel.streamlines.tractogram import LazyTractogram
import numpy as np

from scilpy.reconst.utils import (find_order_from_nb_coeff,
                                  get_b_matrix, get_maximas)
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_sphere_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.tools import get_theta
from scilpy.tracking.utils import (add_mandatory_options_tracking,
                                   add_out_options, add_seeding_options,
                                   add_tracking_options,
                                   verify_streamline_length_options,
                                   verify_seed_options)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    add_mandatory_options_tracking(p)

    track_g = add_tracking_options(p)
    track_g.add_argument('--algo', default='prob',
                         choices=['det', 'prob', 'eudx'],
                         help='Algorithm to use. [%(default)s]')
    add_sphere_arg(track_g, symmetric_only=True)

    add_seeding_options(p)
    out_g = add_out_options(p)

    out_g.add_argument('--seed', type=int,
                       help='Random number generator seed.')

    log_g = p.add_argument_group('Logging options')
    add_verbose_arg(log_g)

    return p


def _get_direction_getter(args):
    odf_data = nib.load(args.in_odf).get_fdata(dtype=np.float32)
    sphere = HemiSphere.from_sphere(get_sphere(args.sphere))
    theta = get_theta(args.theta, args.algo)

    non_zeros_count = np.count_nonzero(np.sum(odf_data, axis=-1))
    non_first_val_count = np.count_nonzero(np.argmax(odf_data, axis=-1))

    if args.algo in ['det', 'prob']:
        if non_first_val_count / non_zeros_count > 0.5:
            logging.warning('Input detected as peaks. Input should be'
                            'fodf for det/prob, verify input just in case.')
        if args.algo == 'det':
            dg_class = DeterministicMaximumDirectionGetter
        else:
            dg_class = ProbabilisticDirectionGetter
        return dg_class.from_shcoeff(
            shcoeff=odf_data, max_angle=theta, sphere=sphere,
            basis_type=args.sh_basis,
            relative_peak_threshold=args.sf_threshold)
    elif args.algo == 'eudx':
        # Code for type EUDX. We don't use peaks_from_model
        # because we want the peaks from the provided sh.
        odf_shape_3d = odf_data.shape[:-1]
        dg = PeaksAndMetrics()
        dg.sphere = sphere
        dg.ang_thr = theta
        dg.qa_thr = args.sf_threshold

        # Heuristic to find out if the input are peaks or fodf
        # fodf are always around 0.15 and peaks around 0.75
        if non_first_val_count / non_zeros_count > 0.5:
            logging.info('Input detected as peaks.')
            nb_peaks = odf_data.shape[-1] // 3
            slices = np.arange(0, 15+1, 3)
            peak_values = np.zeros(odf_shape_3d+(nb_peaks,))
            peak_indices = np.zeros(odf_shape_3d+(nb_peaks,))

            for idx in np.argwhere(np.sum(odf_data, axis=-1)):
                idx = tuple(idx)
                for i in range(nb_peaks):
                    peak_values[idx][i] = np.linalg.norm(
                        odf_data[idx][slices[i]:slices[i+1]], axis=-1)
                    peak_indices[idx][i] = sphere.find_closest(
                        odf_data[idx][slices[i]:slices[i+1]])

            dg.peak_dirs = odf_data
        else:
            logging.info('Input detected as fodf.')
            npeaks = 5
            peak_dirs = np.zeros((odf_shape_3d + (npeaks, 3)))
            peak_values = np.zeros((odf_shape_3d + (npeaks, )))
            peak_indices = np.full((odf_shape_3d + (npeaks, )), -1,
                                   dtype='int')
            b_matrix = get_b_matrix(
                find_order_from_nb_coeff(odf_data), sphere, args.sh_basis)

            for idx in np.argwhere(np.sum(odf_data, axis=-1)):
                idx = tuple(idx)
                directions, values, indices = get_maximas(odf_data[idx],
                                                          sphere, b_matrix,
                                                          args.sf_threshold, 0)
                if values.shape[0] != 0:
                    n = min(npeaks, values.shape[0])
                    peak_dirs[idx][:n] = directions[:n]
                    peak_values[idx][:n] = values[:n]
                    peak_indices[idx][:n] = indices[:n]

            dg.peak_dirs = peak_dirs

        dg.peak_values = peak_values
        dg.peak_indices = peak_indices

        return dg


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    assert_inputs_exist(parser, [args.in_odf, args.in_seed, args.in_mask])
    assert_outputs_exist(parser, args, args.out_tractogram)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress)
    verify_seed_options(parser, args)

    mask_img = nib.load(args.in_mask)
    mask_data = get_data_as_mask(mask_img, dtype=bool)

    # Make sure the data is isotropic. Else, the strategy used
    # when providing information to dipy (i.e. working as if in voxel space)
    # will not yield correct results.
    odf_sh_img = nib.load(args.in_odf)
    if not np.allclose(np.mean(odf_sh_img.header.get_zooms()[:3]),
                       odf_sh_img.header.get_zooms()[0], atol=1e-03):
        parser.error(
            'ODF SH file is not isotropic. Tracking cannot be ran robustly.')

    if args.npv:
        nb_seeds = args.npv
        seed_per_vox = True
    elif args.nt:
        nb_seeds = args.nt
        seed_per_vox = False
    else:
        nb_seeds = 1
        seed_per_vox = True

    voxel_size = odf_sh_img.header.get_zooms()[0]
    vox_step_size = args.step_size / voxel_size
    seed_img = nib.load(args.in_seed)

    if np.count_nonzero(seed_img.get_fdata(dtype=np.float32)) == 0:
        raise IOError('The image {} is empty. '
                      'It can\'t be loaded as '
                      'seeding mask.'.format(args.in_seed))

    # Note. Seeds are in voxel world, center origin.
    # (See the examples in random_seeds_from_mask).
    seeds = track_utils.random_seeds_from_mask(
        seed_img.get_fdata(dtype=np.float32),
        np.eye(4),
        seeds_count=nb_seeds,
        seed_count_per_voxel=seed_per_vox,
        random_seed=args.seed)

    # Tracking is performed in voxel space
    max_steps = int(args.max_length / args.step_size) + 1
    streamlines_generator = LocalTracking(
        _get_direction_getter(args),
        BinaryStoppingCriterion(mask_data),
        seeds, np.eye(4),
        step_size=vox_step_size, max_cross=1,
        maxlen=max_steps,
        fixedstep=True, return_all=True,
        random_seed=args.seed,
        save_seeds=args.save_seeds)

    scaled_min_length = args.min_length / voxel_size
    scaled_max_length = args.max_length / voxel_size

    if args.save_seeds:
        filtered_streamlines, seeds = \
            zip(*((s, p) for s, p in streamlines_generator
                  if scaled_min_length <= length(s) <= scaled_max_length))
        data_per_streamlines = {'seeds': lambda: seeds}
    else:
        filtered_streamlines = \
            (s for s in streamlines_generator
             if scaled_min_length <= length(s) <= scaled_max_length)
        data_per_streamlines = {}

    if args.compress:
        # Compressing. Threshold is in mm, but we are working in voxel space.
        # Equivalent of sft.to_voxmm:  streamline *= voxres
        # Equivalent of sft.to_vox: streamline /= voxres
        voxres = np.asarray(odf_sh_img.header.get_zooms()[0:3])
        filtered_streamlines = (
            compress_streamlines(s * voxres,
                                 args.compress) / voxres
            for s in filtered_streamlines)

    tractogram = LazyTractogram(lambda: filtered_streamlines,
                                data_per_streamlines,
                                affine_to_rasmm=seed_img.affine)

    filetype = nib.streamlines.detect_format(args.out_tractogram)
    reference = get_reference_info(seed_img)
    header = create_tractogram_header(filetype, *reference)

    # Use generator to save the streamlines on-the-fly
    nib.streamlines.save(tractogram, args.out_tractogram, header=header)


if __name__ == '__main__':
    main()
