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

It is possible to use the ball-and-stick model for single-shell and multi-shell
data. In this case, the peak file is not mandatory. Multi-shell should follow a
"NODDI protocol" (low and high b-values), multiple shells with similar b-values
should not be used with COMMIT.

The output from COMMIT is:
- fit_NRMSE.nii.gz
    fiting error (Normalized Root Mean Square Error)
- fit_RMSE.nii.gz
    fiting error (Root Mean Square Error)
- results.pickle
    Dictionary containing the experiment parameters and final weights
- compartment_EC.nii.gz (est. Extra-Cellular signal fraction)
- compartment_IC.nii.gz (est. Intra-Cellular signal fraction)
- compartment_ISO.nii.gz (est. isotropic signal fraction
  (freewater comportment))
    Each of COMMIT compartments
- streamline_weights.txt
    Text file containing the commit weights for each streamline of the
    input tractogram.
- streamlines_length.txt
    Text file containing the length (mm) of each streamline
- tot_streamline_weights
    Text file containing the total commit weights of each streamline.
    Equal to commit_weights * streamlines_length (W_i * L_i)
- essential.trk / non_essential.trk
    Tractograms containing the streamlines below or equal (essential) and
    above (non_essential) a threshold_weights of 0.

This script can divide the input tractogram in two using a threshold to apply
on the streamlines' weight. The threshold used is 0.0, keeping only
streamlines that have non-zero weight and that contribute to explain the DWI
signal. Streamlines with 0 weight are essentially not necessary according to
COMMIT.

COMMIT2 is available only for HDF5 data from
scil_tractogram_segment_bundles_for_connectivity.py and
with the --ball_stick option. Use the --commit2 option to activite it, slightly
longer computation time. This wrapper offers a simplify way to call COMMIT,
but does not allow to use (or fine-tune) every parameters. If you want to use
COMMIT with full access to all parameters,
visit: https://github.com/daducci/COMMIT

When tunning parameters, such as --iso_diff, --para_diff, --perp_diff or
--lambda_commit_2 you should evaluate the quality of results by:
    - Looking at the 'density' (GTM) of the connnectome (essential tractogram)
    - Confirm the quality of WM bundles reconstruction (essential tractogram)
    - Inspect the (N)RMSE map and look for peaks or anomalies
    - Compare the density map before and after (essential tractogram)

Formerly: scil_run_commit.py
"""

import argparse
from contextlib import redirect_stdout
import io
import logging
import os
import shutil
import sys
import tempfile

import commit
from commit import trk2dictionary
from dipy.io.stateful_tractogram import (Origin, Space,
                                         StatefulTractogram)
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.io.utils import is_header_compatible
from dipy.io.gradients import read_bvals_bvecs
from dipy.tracking.streamlinespeed import length
import h5py
import numpy as np
import nibabel as nib

from scilpy.io.gradients import fsl2mrtrix
from scilpy.io.hdf5 import reconstruct_streamlines_from_hdf5
from scilpy.io.streamlines import reconstruct_streamlines
from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             redirect_stdout_c)
from scilpy.gradients.bvec_bval_tools import identify_shells


EPILOG = """
References:
[1] Daducci, Alessandro, et al. "COMMIT: convex optimization modeling for
    microstructure informed tractography." IEEE transactions on medical
    imaging 34.1 (2014): 246-257.
[2] Schiavi, Simona, et al. "A new method for accurate in vivo mapping of
    human brain connections using microstructural and anatomical information."
    Science advances 6.31 (2020): eaba8245.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
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
    p.add_argument('--nbr_iter', type=int, default=1000,
                   help='Maximum number of iterations [%(default)s].')
    p.add_argument('--in_peaks',
                   help='Peaks file representing principal direction(s) '
                        'locally,\n typically coming from fODFs. This file is '
                        'mandatory for the default\n stick-zeppelin-ball '
                        'model.')
    p.add_argument('--in_tracking_mask',
                   help='Binary mask where tratography was allowed.\n'
                        'If not set, uses a binary mask computed from '
                        'the streamlines.')

    g0 = p.add_argument_group(title='COMMIT2 options')
    g0.add_argument('--commit2', action='store_true',
                    help='Run commit2, requires .h5 as input and will force\n'
                         'ball&stick model.')
    g0.add_argument('--lambda_commit_2', type=float, default=1e-3,
                    help='Specify the clustering prior strength '
                    '[%(default)s].')

    g1 = p.add_argument_group(title='Model options')
    g1.add_argument('--ball_stick', action='store_true',
                    help='Use the ball&Stick model, disable the zeppelin '
                         'compartment.\nOnly model suitable for single-shell '
                         'data.')
    g1.add_argument('--para_diff', type=float,
                    help='Parallel diffusivity in mm^2/s.\n'
                         'Default for ball_stick: 1.7E-3\n'
                         'Default for stick_zeppelin_ball: 1.7E-3')
    g1.add_argument('--perp_diff', nargs='+', type=float,
                    help='Perpendicular diffusivity in mm^2/s.\n'
                         'Default for ball_stick: None\n'
                         'Default for stick_zeppelin_ball: [0.51E-3]')
    g1.add_argument('--iso_diff', nargs='+', type=float,
                    help='Istropic diffusivity in mm^2/s.\n'
                         'Default for ball_stick: [2.0E-3]\n'
                         'Default for stick_zeppelin_ball: [1.7E-3, 3.0E-3]')

    g2 = p.add_argument_group(title='Tractogram options')
    g2.add_argument('--keep_whole_tractogram', action='store_true',
                    help='Save a tractogram copy with streamlines weights in '
                         'the data_per_streamline\n[%(default)s].')

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
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def _save_results_wrapper(args, tmp_dir, ext, hdf5_file, offsets_list,
                          sub_dir, is_commit_2):
    out_dir = os.path.join(args.out_dir, sub_dir)
    os.mkdir(out_dir)
    # Simplifying output for streamlines and cleaning output directory
    commit_results_dir = os.path.join(tmp_dir.name,
                                      'Results_StickZeppelinBall')
    streamline_weights = np.loadtxt(os.path.join(commit_results_dir,
                                                 'streamline_weights.txt'))

    sft = load_tractogram(args.in_tractogram, 'same')
    length_list = length(sft.streamlines)
    np.savetxt(os.path.join(commit_results_dir, 'streamlines_length.txt'),
               length_list)
    np.savetxt(os.path.join(commit_results_dir,
                            'streamline_weights_by_length.txt'),
               streamline_weights*length_list)

    if ext == '.h5':
        new_filename = os.path.join(commit_results_dir,
                                    'decompose_commit.h5')
        with h5py.File(new_filename, 'w') as new_hdf5_file:
            new_hdf5_file.attrs['affine'] = sft.affine
            new_hdf5_file.attrs['dimensions'] = sft.dimensions
            new_hdf5_file.attrs['voxel_sizes'] = sft.voxel_sizes
            new_hdf5_file.attrs['voxel_order'] = sft.voxel_order
            # Assign the weights into the hdf5, while respecting
            # the ordering of connections/streamlines
            logging.info('Adding commit weights to {}.'.format(new_filename))
            for i, key in enumerate(list(hdf5_file.keys())):
                new_group = new_hdf5_file.create_group(key)
                old_group = hdf5_file[key]
                tmp_streamline_weights = \
                    streamline_weights[offsets_list[i]:offsets_list[i+1]]

                essential_ind = np.where(tmp_streamline_weights > 0)[0]
                tmp_streamline_weights = tmp_streamline_weights[essential_ind]

                tmp_streamlines = reconstruct_streamlines(
                    old_group['data'],
                    old_group['offsets'],
                    old_group['lengths'],
                    indices=essential_ind)
                tmp_length_list = length(tmp_streamlines)
                # Replacing the data with the one above the threshold
                # Safe since this hdf5 was a copy in the first place
                new_group.create_dataset('data',
                                         data=tmp_streamlines.get_data(),
                                         dtype=np.float32)
                new_group.create_dataset('offsets',
                                         data=tmp_streamlines._offsets,
                                         dtype=np.int64)
                new_group.create_dataset('lengths',
                                         data=tmp_streamlines._lengths,
                                         dtype=np.int32)

                for dps_key in hdf5_file[key].keys():
                    if dps_key not in ['data', 'offsets', 'lengths']:
                        new_group.create_dataset(
                            key, data=hdf5_file[key][dps_key][essential_ind])

                dps_key = 'commit2_weights' if is_commit_2 else \
                    'commit1_weights'
                dps_key_tot = 'tot_commit2_weights' if is_commit_2 else \
                    'tot_commit1_weights'
                new_group.create_dataset(dps_key,
                                         data=tmp_streamline_weights)
                new_group.create_dataset(
                    dps_key_tot,
                    data=tmp_streamline_weights*tmp_length_list)

    files = os.listdir(commit_results_dir)
    for f in files:
        shutil.copy(os.path.join(commit_results_dir, f), out_dir)

    dps_key = 'commit2_weights' if is_commit_2 else \
        'commit1_weights'
    dps_key_tot = 'tot_commit2_weights' if is_commit_2 else \
        'tot_commit1_weights'
    # Reload is needed because of COMMIT handling its file by itself
    sft.data_per_streamline[dps_key] = streamline_weights
    sft.data_per_streamline[dps_key_tot] = streamline_weights*length_list

    essential_ind = np.where(streamline_weights > 0)[0]
    nonessential_ind = np.where(streamline_weights <= 0)[0]
    logging.info('{} essential streamlines were kept at'.format(
        len(essential_ind)))
    logging.info('{} nonessential streamlines were kept'.format(
        len(nonessential_ind)))

    save_tractogram(sft[essential_ind],
                    os.path.join(out_dir,
                    'essential_tractogram.trk'))
    save_tractogram(sft[nonessential_ind],
                    os.path.join(out_dir,
                    'nonessential_tractogram.trk'))
    if args.keep_whole_tractogram:
        output_filename = os.path.join(out_dir, 'tractogram.trk')
        logging.info('Saving tractogram with weights as {}'.format(
            output_filename))
        save_tractogram(sft, output_filename)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    # COMMIT has some c-level stdout and non-logging print that cannot
    # be easily stopped. Manual redirection of all printed output
    if args.verbose == "WARNING":
        f = io.StringIO()
        redirected_stdout = redirect_stdout(f)
        redirect_stdout_c() 
    else:
        logging.getLogger().setLevel(logging.getLevelName(args.verbose))
        redirected_stdout = redirect_stdout(sys.stdout)

    assert_inputs_exist(parser, [args.in_tractogram, args.in_dwi,
                                 args.in_bval, args.in_bvec],
                        [args.in_peaks, args.in_tracking_mask])
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       optional=args.save_kernels)

    if args.commit2:
        if os.path.splitext(args.in_tractogram)[1] != '.h5':
            parser.error('COMMIT2 requires .h5 file for connectomics.')
        args.ball_stick = True

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

    tmp_dir = tempfile.TemporaryDirectory()
    hdf5_file = None
    offsets_list = None
    if ext == '.h5':
        logging.info('Reconstructing {} into a tractogram for COMMIT.'.format(
            args.in_tractogram))

        hdf5_file = h5py.File(args.in_tractogram, 'r')
        if not (np.allclose(hdf5_file.attrs['affine'], dwi_img.affine,
                            atol=1e-03)
                and np.array_equal(hdf5_file.attrs['dimensions'],
                                   dwi_img.shape[0:3])):
            parser.error('{} does not have a compatible header with {}'.format(
                args.in_tractogram, args.in_dwi))

        # Keep track of the order of connections/streamlines in relation to the
        # tractogram as well as the number of streamlines for each connection.
        bundle_groups_len = []
        hdf5_keys = list(hdf5_file.keys())
        streamlines = []
        for key in hdf5_keys:
            tmp_streamlines = reconstruct_streamlines_from_hdf5(hdf5_file[key])
            streamlines.extend(tmp_streamlines)
            bundle_groups_len.append(len(tmp_streamlines))

        offsets_list = np.cumsum([0]+bundle_groups_len)
        sft = StatefulTractogram(streamlines, args.in_dwi,
                                 Space.VOX, origin=Origin.TRACKVIS)
        tmp_tractogram_filename = os.path.join(tmp_dir.name, 'tractogram.trk')

        # Keeping the input variable, saving trk file for COMMIT internal use
        save_tractogram(sft, tmp_tractogram_filename)
        args.in_tractogram = tmp_tractogram_filename

    # Writing the scheme file with proper shells
    tmp_scheme_filename = os.path.join(tmp_dir.name, 'gradients.b')
    tmp_bval_filename = os.path.join(tmp_dir.name, 'bval')
    bvals, _ = read_bvals_bvecs(args.in_bval, args.in_bvec)
    shells_centroids, indices_shells = identify_shells(bvals, args.b_thr,
                                                       round_centroids=True)
    np.savetxt(tmp_bval_filename, shells_centroids[indices_shells],
               newline=' ', fmt='%i')
    fsl2mrtrix(tmp_bval_filename, args.in_bvec, tmp_scheme_filename)
    logging.info('Lauching COMMIT on {} shells at found at {}.'.format(
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
                           path_out=tmp_dir.name)

        # Preparation for fitting
        commit.core.setup()
        mit = commit.Evaluation('.', '.')

        # FIX for very small values during HCP processing
        # (based on order of magnitude of signal)
        img = nib.load(args.in_dwi)
        data = img.get_fdata(dtype=np.float32)
        data[data <
             (0.001*10**np.floor(np.log10(np.mean(data[data > 0]))))] = 0
        nib.save(nib.Nifti1Image(data, img.affine),
                 os.path.join(tmp_dir.name, 'dwi_zero_fix.nii.gz'))

        mit.load_data(os.path.join(tmp_dir.name, 'dwi_zero_fix.nii.gz'),
                      tmp_scheme_filename)
        mit.set_model('StickZeppelinBall')

        if args.ball_stick:
            logging.info('Disabled zeppelin, using the Ball & Stick model.')
            para_diff = args.para_diff or 1.7E-3
            perp_diff = []
            isotropc_diff = args.iso_diff or [2.0E-3]
            mit.model.set(para_diff, perp_diff, isotropc_diff)
        else:
            logging.info('Using the Stick Zeppelin Ball model.')
            para_diff = args.para_diff or 1.7E-3
            perp_diff = args.perp_diff or [0.85E-3, 0.51E-3]
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

        mit.generate_kernels(ndirs=args.nbr_dir, regenerate=regenerate_kernels)
        if args.compute_only:
            return
        mit.load_kernels()
        use_mask = args.in_tracking_mask is not None
        mit.load_dictionary(tmp_dir.name,
                            use_all_voxels_in_mask=use_mask)
        mit.set_threads(args.nbr_processes)

        mit.build_operator(build_dir=os.path.join(tmp_dir.name, 'build/'))
        tol_fun = 1e-2 if args.commit2 else 1e-3
        mit.fit(tol_fun=tol_fun, max_iter=args.nbr_iter, verbose=False)
        mit.save_results()
        _save_results_wrapper(args, tmp_dir, ext, hdf5_file, offsets_list,
                              'commit_1/', False)

        if args.commit2:
            tmp = np.insert(np.cumsum(bundle_groups_len), 0, 0)
            group_idx = np.array([np.arange(tmp[i], tmp[i+1])
                                  for i in range(len(tmp)-1)])
            group_w = np.empty_like(bundle_groups_len, dtype=np.float64)
            for k in range(len(bundle_groups_len)):
                group_w[k] = np.sqrt(bundle_groups_len[k]) / \
                    (np.linalg.norm(mit.x[group_idx[k]]) + 1e-12)
            prior_on_bundles = commit.solvers.init_regularisation(
                mit, structureIC=group_idx, weightsIC=group_w,
                regnorms=[commit.solvers.group_sparsity,
                          commit.solvers.non_negative,
                          commit.solvers.non_negative],
                lambdas=[args.lambda_commit_2, 0.0, 0.0])
            mit.fit(tol_fun=1e-3, max_iter=args.nbr_iter,
                    regularisation=prior_on_bundles, verbose=False)
            mit.save_results()
            _save_results_wrapper(args, tmp_dir, ext, hdf5_file, offsets_list,
                                  'commit_2/', True)

    tmp_dir.cleanup()


if __name__ == "__main__":
    main()
