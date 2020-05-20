#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Estimate the fit between a provided tractogram and DWI. Assign a weight to each
streamline that represent how well they explain the signal.
Default is stick-zeppelin-ball for multi-shells data.

The real output from COMMIT is:
- fit_NRMSE.nii.gz
    fiting error (Normalized Root Mean Square Error)
- fit_RMSE.nii.gz
    fiting error (Root Mean Square Error)
- results.pickle
    Dictionary containing the experiment parameters and final weights
- compartment_EC.nii.gz
- compartment_IC.nii.gz
- compartment_ISO.nii.gz
    Each of COMMIT compartments

This script can divide the input tractogram in two using a threshold to apply
on the streamlines' weight.
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
                   help='Diffusion from which the fodf were computed.')
    p.add_argument('in_bvals',
                   help='Bvals in the FSL format.')
    p.add_argument('in_bvecs',
                   help='Bvecs in the FSL format..')
    p.add_argument('in_peaks',
                   help='Peaks extracted from the fodf.')
    p.add_argument('out_dir',
                   help='Output directory for the COMMIT maps.')

    p.add_argument('--tracking_mask',
                   help='Binary mask were tratography was allowed.\n'
                        'If not set, uses a binary mask computed from '
                        'the streamlines.')

    g1 = p.add_argument_group(title='Model options')
    g1.add_argument('--ball_stick', action='store_true',
                    help='Use the ball&Stick model.\n'
                         'Disable the zeppelin compartment for single-shell.')
    g1.add_argument('--parallel_diff', type=float,
                    help='Parallel diffusivity in mm^2/s.\n'
                    'Default for ball_stick: 1.7E-3\n'
                    'Default for stick_zeppelin_ball: 1.7E-3')
    g1.add_argument('--perpendicular_diff', nargs='+', type=float,
                    help='Perpendicular diffusivity in mm^2/s.\n'
                    'Default for ball_stick: None\n'
                    'Default for stick_zeppelin_ball: '
                    '[1.19E-3, 0.85E-3, 0.51E-3, 0.17E-3]')
    g1.add_argument('--isotropic_diff', nargs='+', type=float,
                    help='Istropic diffusivity in mm^2/s.\n'
                    'Default for ball_stick: 2.0E-3\n'
                    'Default for stick_zeppelin_ball: [1.7E-3, 3.0E-3]')

    g2 = p.add_argument_group(title='Tractogram options')
    g2.add_argument('--assign_weights', action='store_true',
                    help='Store the streamlines weights in the '
                    'data_per_streamline.')
    g2.add_argument('--threshold_weights', type=float, metavar='THRESHOLD',
                    help='Split the tractogram in two. Valid and invalid, '
                    'based on the provided threshold.')

    g3 = p.add_argument_group(title='Kernels options')
    kern = g3.add_mutually_exclusive_group()
    kern.add_argument('--save_kernels', metavar='DIRECTORY',
                      help='Output directory for the COMMIT kernels.')
    kern.add_argument('--load_kernels', metavar='DIRECTORY',
                      help='Input directory where the COMMIT kernels are '
                           'located.')
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
                                 args.in_bvals, args.in_bvecs,
                                 args.in_peaks], args.tracking_mask)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       optional=args.save_kernels)
    if args.load_kernels and not os.path.isdir(args.load_kernels):
        parser.error('Kernels directory does not exist.')

    if args.ball_stick and args.perpendicular_diff:
        parser.error('Cannot use --perpendicular_diff with ball&stick.')

    if args.ball_stick and args.isotropic_diff and len(args.isotropic_diff) > 1:
        parser.error(
            'Cannot use more than one --isotropic_diff with ball&stick.')

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
    bvals, bvecs = read_bvals_bvecs(args.in_bvals, args.in_bvecs)
    shells_centroids, _ = identify_shells(bvals)
    fsl2mrtrix(args.in_bvals, args.in_bvecs, tmp_scheme_filename)
    logging.debug('Lauching COMMIT on {} shells at found at {}.'.format(
        len(shells_centroids),
        shells_centroids))

    if len(shells_centroids) == 2 and not args.disable_zeppelin:
        parser.error('The DWI data appears to be single-shell.\n'
                     'Use --disable_zeppelin for single-shell.')

    with redirected_stdout:
        # Setting up the tractogram and nifti files
        trk2dictionary.run(filename_tractogram=args.in_tractogram,
                           filename_peaks=args.in_peaks,
                           peaks_use_affine=False,
                           filename_mask=args.tracking_mask,
                           ndirs=500,
                           gen_trk=False,
                           path_out=tmp_dir.name)

        # Preparation for fitting
        commit.core.setup(ndirs=500)
        mit = commit.Evaluation('.', '.')
        mit.load_data(args.in_dwi, tmp_scheme_filename)
        mit.set_model('StickZeppelinBall')

        if args.ball_stick:
            logging.debug('Disabled zeppelin, using the Ball & Stick model.')
            parallel_diff = args.parallel_diff or 1.7E-3
            perpendicular_diff = []
            isotropc_diff = args.isotropic_diff or [2.0E-3]
            mit.model.set(parallel_diff, perpendicular_diff, isotropc_diff)
        else:
            logging.debug('Using the Stick Zeppelin Ball model.')
            parallel_diff = args.parallel_diff or 1.7E-3
            perpendicular_diff = args.perpendicular_diff or \
                [1.19E-3, 0.85E-3, 0.51E-3, 0.17E-3]
            isotropc_diff = args.isotropic_diff or [1.7E-3, 3.0E-3]
            mit.model.set(parallel_diff, perpendicular_diff, isotropc_diff)

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
        mit.load_kernels()
        mit.load_dictionary(tmp_dir.name)
        mit.set_threads(args.nbr_processes)
        mit.build_operator()
        mit.fit(tol_fun=1e-3, max_iter=500, verbose=0)
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

    # Save split tractogram (valid/invalid) and/or saving the tractogram with
    # data_per_streamline updated
    if args.assign_weights or args.threshold_weights:
        # Reload is needed because of COMMIT handling its file by itself
        tractogram_file = nib.streamlines.load(args.in_tractogram)
        tractogram = tractogram_file.tractogram
        tractogram.data_per_streamline['commit_weights'] = commit_weights

        if args.threshold_weights:
            valid_ind = np.where(
                commit_weights >= args.threshold_weights)[0]
            invalid_ind = np.where(
                commit_weights < args.threshold_weights)[0]
            logging.debug('{} valid streamlines were kept at threshold {}'.format(
                len(valid_ind), args.threshold_weights))
            logging.debug('{} invalid streamlines were kept at threshold {}'.format(
                len(invalid_ind), args.threshold_weights))

            valid_streamlines = tractogram.streamlines[valid_ind]
            valid_data_per_streamline = tractogram.data_per_streamline[valid_ind]
            valid_data_per_point = tractogram.data_per_point[valid_ind]
            valid_tractogram = Tractogram(valid_streamlines,
                                          data_per_point=valid_data_per_point,
                                          data_per_streamline=valid_data_per_streamline,
                                          affine_to_rasmm=np.eye(4))

            invalid_streamlines = tractogram.streamlines[invalid_ind]
            invalid_data_per_streamline = tractogram.data_per_streamline[invalid_ind]
            invalid_data_per_point = tractogram.data_per_point[invalid_ind]
            invalid_tractogram = Tractogram(invalid_streamlines,
                                            data_per_point=invalid_data_per_point,
                                            data_per_streamline=invalid_data_per_streamline,
                                            affine_to_rasmm=np.eye(4))

            nib.streamlines.save(valid_tractogram,
                                 os.path.join(args.out_dir,
                                              'valid_tractogram.trk'))
            nib.streamlines.save(invalid_tractogram,
                                 os.path.join(args.out_dir,
                                              'invalid_tractogram.trk'))
        if args.assign_weights:
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
