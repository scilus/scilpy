#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


import random
from scilpy.utils.bvec_bval_tools import identify_shells
import sys
import io
from contextlib import contextmanager
from contextlib import redirect_stdout
import tempfile
import shutil
from commit import trk2dictionary
import amico
import commit
import argparse
import os
import logging
import pickle

import numpy as np
import nibabel as nib
from nibabel.streamlines import Tractogram
from fury import window, actor, interactor
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible
from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty)
from scilpy.utils.bvec_bval_tools import fsl2mrtrix
from scilpy.io.streamlines import lazy_streamlines_count


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='.')
    p.add_argument('in_dwi',
                   help='.')
    p.add_argument('in_bvals',
                   help='.')
    p.add_argument('in_bvecs',
                   help='.')
    p.add_argument('in_peaks',
                   help='.')
    p.add_argument('out_dir',
                   help='.')

    p.add_argument('--tracking_mask',
                   help='If not set, use a binary mask computed from the streamlines.')
    p.add_argument('--disable_zeppelin', action='store_true',
                   help='.')

    p.add_argument('--assign_weights', action='store_true',
                   help='Store the streamlines weights in the data_per_streamline.')
    p.add_argument('--threshold_weights', type=float, metavar='THRESHOLD',
                   help='Split the tractogram in two. Valid and invalid, based '
                        'on the provided threshold.')

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
                                       create_dir=True)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        redirected = redirect_stdout(sys.stdout)
    else:
        f = io.StringIO()
        redirected = redirect_stdout(f)
        redirect_stdout_c()

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_scheme_filename = os.path.join(tmp_dir.name, 'gradients.scheme')
    bvals, bvecs = read_bvals_bvecs(args.in_bvals, args.in_bvecs)
    shells_centroids, _ = identify_shells(bvals)
    fsl2mrtrix(args.in_bvals, args.in_bvecs, tmp_scheme_filename)
    logging.debug('Lauching COMMIT on {} shells at found at {}.'.format(
        len(shells_centroids),
        shells_centroids))

    with redirected:
        trk2dictionary.run(filename_tractogram=args.in_tractogram,
                           filename_peaks=args.in_peaks,
                           peaks_use_affine=False,
                           filename_mask=args.tracking_mask,
                           ndirs=500,
                           gen_trk=False,
                           path_out=tmp_dir.name)

        # preparation and fitting
        commit.core.setup(ndirs=500)
        mit = commit.Evaluation('.', '.')
        mit.load_data(args.in_dwi, tmp_scheme_filename)
        mit.set_model('StickZeppelinBall')
        if args.disable_zeppelin:
            logging.debug('Disabled zeppelin, using the Ball & Sticks model.')
            zeppelin_priors = []
        else:
            logging.debug('Using the Stick Zeppelin Ball model.')
            zeppelin_priors = [0.7]
        mit.model.set(1.7E-3, zeppelin_priors, [2.0E-3])
        mit.set_config('ATOMS_path', os.path.join(tmp_dir.name,
                                                  'kernels',
                                                  mit.model.id))
        mit.generate_kernels(ndirs=500, regenerate=True)
        mit.load_kernels()
        mit.load_dictionary(tmp_dir.name)
        mit.set_threads(args.nbr_processes)
        mit.build_operator()
        mit.fit(tol_fun=1e-3, max_iter=500, verbose=0)
        mit.save_results()

    # Simplifying output for streamlines and cleaning output directory
    commit_results_dir = os.path.join(tmp_dir.name,
                                      'Results_StickZeppelinBall')
    file = open(os.path.join(commit_results_dir, 'results.pickle'), 'rb')
    commit_output_dict = pickle.load(file)
    nbr_streamlines = lazy_streamlines_count(args.in_tractogram)
    commit_weights = commit_output_dict[2][:nbr_streamlines]
    np.savetxt(os.path.join(commit_results_dir, 'streamlines_weights.txt'),
               commit_weights)
    dest = shutil.move(commit_results_dir, args.out_dir)
    tmp_dir.cleanup()

    # Save split tractogram (valid/invalid) and/or saving the tractogram with
    # data_per_streamline updated
    if args.assign_weights or args.threshold_weights:
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

            nib.streamlines.save(valid_tractogram, os.path.join(args.out_dir,
                                                                'valid_tractogram.trk'))
            nib.streamlines.save(invalid_tractogram, os.path.join(args.out_dir,
                                                                  'invalid_tractogram.trk'))
        if args.assign_weights:
            output_filename = os.path.join(args.out_dir, 'tractogram.trk')
            logging.debug('Saving tractogram with weights as {}'.format(
                output_filename))
            nib.streamlines.save(tractogram_file, output_filename)


if __name__ == "__main__":
    main()
