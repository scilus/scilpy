#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_commit.py', '--help')
    assert ret.success


def test_execution_commit_amico(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tracking = os.path.join(get_home(), 'commit_amico',
                               'tracking.trk')
    in_dwi = os.path.join(get_home(), 'commit_amico',
                          'dwi.nii.gz')
    in_bval = os.path.join(get_home(), 'commit_amico',
                           'dwi.bval')
    in_bvec = os.path.join(get_home(), 'commit_amico',
                           'dwi.bvec')
    in_mask = os.path.join(get_home(), 'commit_amico',
                           'mask.nii.gz')
    in_peaks = os.path.join(get_home(), 'commit_amico',
                            'peaks.nii.gz')
    ret = script_runner.run('scil_tractogram_commit.py', in_tracking, in_dwi,
                            in_bval, in_bvec, 'results_bzs/',
                            '--b_thr', '30', '--nbr_dir', '500',
                            '--nbr_iter', '500', '--in_peaks', in_peaks,
                            '--in_tracking_mask', in_mask,
                            '--para_diff', '1.7E-3',
                            '--perp_diff', '1.19E-3', '0.85E-3', '0.51E-3', '0.17E-3',
                            '--iso_diff', '1.7E-3', '3.0E-3',
                            '--processes', '1')
    assert ret.success
