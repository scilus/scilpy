#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_local_tracking.py',
                            '--help')
    assert ret.success


def test_execution_tracking_fodf(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')
    in_mask = os.path.join(get_home(), 'tracking',
                           'seeding_mask.nii.gz')

    ret = script_runner.run('scil_compute_local_tracking.py', in_fodf,
                            in_mask, in_mask, 'local_prob.trk', '--nt', '1000',
                            '--compress', '0.1', '--sh_basis', 'descoteaux07',
                            '--min_length', '20', '--max_length', '200')
    assert ret.success


def test_execution_tracking_fodf_no_compression(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')
    in_mask = os.path.join(get_home(), 'tracking',
                           'seeding_mask.nii.gz')

    ret = script_runner.run('scil_compute_local_tracking.py', in_fodf,
                            in_mask, in_mask, 'local_prob2.trk',
                            '--nt', '100', '--sh_basis', 'descoteaux07',
                            '--max_length', '200')

    assert ret.success


def test_execution_tracking_peaks(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_peaks = os.path.join(get_home(), 'tracking',
                            'peaks.nii.gz')
    in_mask = os.path.join(get_home(), 'tracking',
                           'seeding_mask.nii.gz')
    ret = script_runner.run('scil_compute_local_tracking.py', in_peaks,
                            in_mask, in_mask, 'local_eudx.trk', '--nt', '1000',
                            '--compress', '0.1', '--sh_basis', 'descoteaux07',
                            '--min_length', '20', '--max_length', '200',
                            '--algo', 'eudx')
    assert ret.success
