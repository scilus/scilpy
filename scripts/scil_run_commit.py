#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convex Optimization Modeling for Microstructure Informed Tractography (COMMIT)
estimates, globally, how a given tractogram explains the DWI in terms of signal
fit, assuming a certain forward microstructure model. It assigns a weight to
each streamline, which represents how well it explains the DWI signal globally.
The default forward microstructure model is stick-zeppelin-ball, which requires
multi-shell data and a peak file (principal fiber directions in each voxel,
typically from a field of fODFs).

It is possible to use the ball-and-stick model for single-shell data. In this
case, the peak file is not mandatory.

The output from COMMIT is:
- fit_NRMSE.nii.gz
    fiting error (Normalized Root Mean Square Error)
- fit_RMSE.nii.gz
    fiting error (Root Mean Square Error)
- results.pickle
    Dictionary containing the experiment parameters and final weights
- compartment_EC.nii.gz (Extra-Cellular)
- compartment_IC.nii.gz (Intra-Cellular)
- compartment_ISO.nii.gz (isotropic volume fraction (freewater comportment))
    Each of COMMIT compartments
- commit_weights.txt
    Text file containing the commit weights for each streamline of the
    input tractogram.
- essential.trk / non_essential.trk
    Tractograms containing the streamlines below or equal (essential) and
    above (non_essential) the --threshold_weights argument.


This script can divide the input tractogram in two using a threshold to apply
on the streamlines' weight. Typically, the threshold should be 0, keeping only
streamlines that have non-zero weight and that contribute to explain the DWI
signal. Streamlines with 0 weight are essentially not necessary according to
COMMIT.
"""

import argparse
from contextlib import redirect_stdout
import io
import logging
import os
import pickle
import shutil
import sys
import tempfile

import commit
from commit import trk2dictionary
from dipy.io.stateful_tractogram import (Origin, Space,
                                         StatefulTractogram)
from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible
from dipy.io.gradients import read_bvals_bvecs
import h5py
import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram

from scilpy.io.streamlines import (lazy_streamlines_count,
                                   reconstruct_streamlines_from_hdf5)
from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.utils.bvec_bval_tools import fsl2mrtrix, identify_shells


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Input tractogram (.trk or .tck or .h5).')
    p.add_argument('in_dwi',
                   help='Diffusion-weighted images used by COMMIT (.nii.gz).')
    p.add_argument('in_bval',
                   help='b-values in the FSL format (.bval).')
    p.add_argument('in_bvec',
                   help='b-vectors in the FSL format (.bvec).')
    p.add_argument('out_dir',
                   help='Output directory for the COMMIT maps.')

    p.add_argument('--b_thr', type=int, default=40,
                   help='Limit value to consider that a b-value is on an '
                        'existing shell.\nAbove this limit, the b-value is '
                        'placed on a new shell. This includes b0s values.')
    p.add_argument('--nbr_dir', type=int, default=500,
                   help='Number of directions, on the half of the sphere,\n'
                        'representing the possible orientations of the '
                        'response functions [%(default)s].')
    p.add_argument('--nbr_iter', type=int, default=500,
                   help='Maximum number of iterations [%(default)s].')
    p.add_argument('--in_peaks',
                   help='Peaks file representing principal direction(s) '
                        'locally,\n typically coming from fODFs. This file is '
                        'mandatory for the default\n stick-zeppelin-ball '
                        'model, when used with multi-shell data.')
    p.add_argument('--in_tracking_mask',
                   help='Binary mask where tratography was allowed.\n'
                        'If not set, uses a binary mask computed from '
                        'the streamlines.')

    g1 = p.add_argument_group(title='Model options')
    g1.add_argument('--ball_stick', action='store_true',
                    help='Use the ball&Stick model.\nDisable '
                         'the zeppelin compartment for single-shell data.')
    g1.add_argument('--para_diff', type=float,
                    help='Parallel diffusivity in mm^2/s.\n'
                         'Default for ball_stick: 1.7E-3\n'
                         'Default for stick_zeppelin_ball: 1.7E-3')
    g1.add_argument('--perp_diff', nargs='+', type=float,
                    help='Perpendicular diffusivity in mm^2/s.\n'
                         'Default for ball_stick: None\n'
                         'Default for stick_zeppelin_ball: '
                         '[1.19E-3, 0.85E-3, 0.51E-3, 0.17E-3]')
    g1.add_argument('--iso_diff', nargs='+', type=float,
                    help='Istropic diffusivity in mm^2/s.\n'
                         'Default for ball_stick: [2.0E-3]\n'
                         'Default for stick_zeppelin_ball: [1.7E-3, 3.0E-3]')

    g2 = p.add_argument_group(title='Tractogram options')
    g2.add_argument('--keep_whole_tractogram', action='store_true',
                    help='Save a tractogram copy with streamlines weights in '
                         'the data_per_streamline\n[%(default)s].')
    g2.add_argument('--threshold_weights', type=float, metavar='THRESHOLD',
                    default=0.,
                    help='Split the tractogram in two; essential and\n'
                         'nonessential, based on the provided threshold '
                         '[%(default)s].\n Use None to skip this step.')

    g3 = p.add_argument_group(title='Kernels options')
    kern = g3.add_mutually_exclusive_group()
    kern.add_argument('--save_kernels', metavar='DIRECTORY',
                      help='Output directory for the COMMIT kernels.')
    kern.add_argument('--load_kernels', metavar='DIRECTORY',
                      help='Input directory where the COMMIT kernels are '
                           'located.')
    g2.add_argument('--compute_only', action='store_true',
                    help='Compute kernels only, --save_kernels must be used.')
    add_processes_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def redirect_stdout_c():
    sys.stdout.flush()
    newstdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.stdout = os.fdopen(newstdout, 'w')


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram, args.in_dwi,
                                 args.in_bval, args.in_bvec],
                        [args.in_peaks, args.in_tracking_mask])
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       optional=args.save_kernels)

    if args.load_kernels and not os.path.isdir(args.load_kernels):
        parser.error('Kernels directory does not exist.')

    if args.compute_only and not args.save_kernels:
        parser.error('--compute_only must be used with --save_kernels.')

    if args.load_kernels and args.save_kernels:
        parser.error('Cannot load and save kernels at the same time.')

    if args.ball_stick and args.perp_diff:
        parser.error('Cannot use --perp_diff with ball&stick.')

    if not args.ball_stick and not args.in_peaks:
        parser.error('Stick Zeppelin Ball model requires --in_peaks')

    if args.ball_stick and args.iso_diff and len(args.iso_diff) > 1:
        parser.error('Cannot use more than one --iso_diff with '
                     'ball&stick.')

    # If it is a trk, check compatibility of header since COMMIT does not do it
    dwi_img = nib.load(args.in_dwi)
    _, ext = os.path.splitext(args.in_tractogram)
    if ext == '.trk' and not is_header_compatible(args.in_tractogram,
                                                  dwi_img):
        parser.error('{} does not have a compatible header with {}'.format(
            args.in_tractogram, args.in_dwi))

    # COMMIT has some c-level stdout and non-logging print that cannot
    # be easily stopped. Manual redirection of all printed output
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        redirected_stdout = redirect_stdout(sys.stdout)
    else:
        f = io.StringIO()
        redirected_stdout = redirect_stdout(f)
        redirect_stdout_c()

    tmp_dir = tempfile.TemporaryDirectory()
    if ext == '.h5':
        logging.debug('Reconstructing {} into a tractogram for COMMIT.'.format(
            args.in_tractogram))
        streamlines = []
        len_list = [0]
        hdf5_file = h5py.File(args.in_tractogram, 'r')
        if not (np.allclose(hdf5_file.attrs['affine'], dwi_img.affine)
                and np.allclose(hdf5_file.attrs['dimensions'], dwi_img.shape[0:3])):
            parser.error('{} does not have a compatible header with {}'.format(
                args.in_tractogram, args.in_dwi))

        # Keep track of the order of connections/streamlines in relation to the
        # tractogram as well as the number of streamlines for each connection.
        hdf5_keys = list(hdf5_file.keys())
        for key in hdf5_keys:
            tmp_streamlines = reconstruct_streamlines_from_hdf5(hdf5_file,
                                                                key)
            len_list.append(len(tmp_streamlines))
            streamlines.extend(tmp_streamlines)
        len_list = np.cumsum(len_list)

        sft = StatefulTractogram(streamlines, args.in_dwi,
                                 Space.VOX, origin=Origin.TRACKVIS)
        tmp_tractogram_filename = os.path.join(tmp_dir.name, 'tractogram.trk')

        # Keeping the input variable, saving trk file for COMMIT internal use
        save_tractogram(sft, tmp_tractogram_filename)
        initial_hdf5_filename = args.in_tractogram
        args.in_tractogram = tmp_tractogram_filename

    tmp_scheme_filename = os.path.join(tmp_dir.name, 'gradients.scheme')
    tmp_bval_filename = os.path.join(tmp_dir.name, 'bval')
    bvals, _ = read_bvals_bvecs(args.in_bval, args.in_bvec)
    shells_centroids, indices_shells = identify_shells(bvals,
                                                       args.b_thr,
                                                       roundCentroids=True)
    np.savetxt(tmp_bval_filename, shells_centroids[indices_shells],
               newline=' ', fmt='%i')
    fsl2mrtrix(tmp_bval_filename, args.in_bvec, tmp_scheme_filename)
    logging.debug('Lauching COMMIT on {} shells at found at {}.'.format(
        len(shells_centroids),
        shells_centroids))

    if len(shells_centroids) == 2 and not args.ball_stick:
        parser.error('The DWI data appears to be single-shell.\n'
                     'Use --ball_stick for single-shell.')

    with redirected_stdout:
        # Setting up the tractogram and nifti files
        trk2dictionary.run(filename_tractogram=args.in_tractogram,
                           filename_peaks=args.in_peaks,
                           peaks_use_affine=False,
                           filename_mask=args.in_tracking_mask,
                           ndirs=args.nbr_dir,
                           gen_trk=False,
                           path_out=tmp_dir.name)

        # Preparation for fitting
        commit.core.setup(ndirs=args.nbr_dir)
        mit = commit.Evaluation('.', '.')
        mit.load_data(args.in_dwi, tmp_scheme_filename)
        mit.set_model('StickZeppelinBall')

        if args.ball_stick:
            logging.debug('Disabled zeppelin, using the Ball & Stick model.')
            para_diff = args.para_diff or 1.7E-3
            perp_diff = []
            isotropc_diff = args.iso_diff or [2.0E-3]
            mit.model.set(para_diff, perp_diff, isotropc_diff)
        else:
            logging.debug('Using the Stick Zeppelin Ball model.')
            para_diff = args.para_diff or 1.7E-3
            perp_diff = args.perp_diff or \
                [1.19E-3, 0.85E-3, 0.51E-3, 0.17E-3]
            isotropc_diff = args.iso_diff or [1.7E-3, 3.0E-3]
            mit.model.set(para_diff, perp_diff, isotropc_diff)

        # The kernels are, by default, set to be in the current directory
        # Depending on the choice, manually change the saving location
        if args.save_kernels:
            kernels_dir = os.path.join(args.save_kernels)
            regenerate_kernels = True
        elif args.load_kernels:
            kernels_dir = os.path.join(args.load_kernels)
            regenerate_kernels = False
        else:
            kernels_dir = os.path.join(tmp_dir.name, 'kernels', mit.model.id)
            regenerate_kernels = True
        mit.set_config('ATOMS_path', kernels_dir)

        mit.generate_kernels(ndirs=500, regenerate=regenerate_kernels)
        if args.compute_only:
            return
        mit.load_kernels()
        mit.load_dictionary(tmp_dir.name,
                            use_mask=args.in_tracking_mask is not None)
        mit.set_threads(args.nbr_processes)

        mit.build_operator(build_dir=tmp_dir.name)
        mit.fit(tol_fun=1e-3, max_iter=args.nbr_iter, verbose=0)
        mit.save_results()

    # Simplifying output for streamlines and cleaning output directory
    commit_results_dir = os.path.join(tmp_dir.name,
                                      'Results_StickZeppelinBall')
    pk_file = open(os.path.join(commit_results_dir, 'results.pickle'), 'rb')
    commit_output_dict = pickle.load(pk_file)
    nbr_streamlines = lazy_streamlines_count(args.in_tractogram)
    commit_weights = commit_output_dict[2][:nbr_streamlines]
    np.savetxt(os.path.join(commit_results_dir,
                            'commit_weights.txt'),
               commit_weights)

    if ext == '.h5':
        new_filename = os.path.join(commit_results_dir,
                                    'decompose_commit.h5')
        shutil.copy(initial_hdf5_filename, new_filename)
        hdf5_file = h5py.File(new_filename, 'a')

        # Assign the weights into the hdf5, while respecting the ordering of
        # connections/streamlines
        logging.debug('Adding commit weights to {}.'.format(new_filename))
        for i, key in enumerate(hdf5_keys):
            group = hdf5_file[key]
            tmp_commit_weights = commit_weights[len_list[i]:len_list[i+1]]
            if 'commit_weights' in group:
                del group['commit_weights']
            group.create_dataset('commit_weights',
                                 data=tmp_commit_weights)

    files = os.listdir(commit_results_dir)
    for f in files:
        shutil.move(os.path.join(commit_results_dir, f), args.out_dir)

    # Save split tractogram (essential/nonessential) and/or saving the
    # tractogram with data_per_streamline updated
    if args.keep_whole_tractogram or args.threshold_weights is not None:
        # Reload is needed because of COMMIT handling its file by itself
        tractogram_file = nib.streamlines.load(args.in_tractogram)
        tractogram = tractogram_file.tractogram
        tractogram.data_per_streamline['commit_weights'] = commit_weights

        if args.threshold_weights is not None:
            essential_ind = np.where(
                commit_weights > args.threshold_weights)[0]
            nonessential_ind = np.where(
                commit_weights <= args.threshold_weights)[0]
            logging.debug('{} essential streamlines were kept at '
                          'threshold {}'.format(len(essential_ind),
                                                args.threshold_weights))
            logging.debug('{} nonessential streamlines were kept at '
                          'threshold {}'.format(len(nonessential_ind),
                                                args.threshold_weights))

            # TODO PR when Dipy 1.2 is out with sft slicing
            essential_streamlines = tractogram.streamlines[essential_ind]
            essential_dps = tractogram.data_per_streamline[essential_ind]
            essential_dpp = tractogram.data_per_point[essential_ind]
            essential_tractogram = Tractogram(essential_streamlines,
                                              data_per_point=essential_dpp,
                                              data_per_streamline=essential_dps,
                                              affine_to_rasmm=np.eye(4))

            nonessential_streamlines = tractogram.streamlines[nonessential_ind]
            nonessential_dps = tractogram.data_per_streamline[nonessential_ind]
            nonessential_dpp = tractogram.data_per_point[nonessential_ind]
            nonessential_tractogram = Tractogram(nonessential_streamlines,
                                                 data_per_point=nonessential_dpp,
                                                 data_per_streamline=nonessential_dps,
                                                 affine_to_rasmm=np.eye(4))

            nib.streamlines.save(essential_tractogram,
                                 os.path.join(args.out_dir,
                                              'essential_tractogram.trk'),
                                 header=tractogram_file.header)
            nib.streamlines.save(nonessential_tractogram,
                                 os.path.join(args.out_dir,
                                              'nonessential_tractogram.trk'),
                                 header=tractogram_file.header,)
        if args.keep_whole_tractogram:
            output_filename = os.path.join(args.out_dir, 'tractogram.trk')
            logging.debug('Saving tractogram with weights as {}'.format(
                output_filename))
            nib.streamlines.save(tractogram_file, output_filename)

    # Cleanup the temporary directory
    if ext == '.h5':
        hdf5_file.close()
    tmp_dir.cleanup()


if __name__ == "__main__":
    main()
