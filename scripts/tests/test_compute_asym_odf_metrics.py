#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_asym_odf_metrics.py', '--help')
    assert ret.success


def test_execution(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking', 'fodf_full.nii.gz')

    # Using a low resolution sphere for peak extraction reduces process time
    ret = script_runner.run('scil_compute_asym_odf_metrics.py', in_fodf,
                            '--sphere', 'repulsion100')
    assert ret.success


def test_assert_not_all(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking', 'fodf_full.nii.gz')

    ret = script_runner.run('scil_compute_asym_odf_metrics.py', in_fodf,
                            '--not_all')
    assert not ret.success


def test_execution_not_all(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking', 'fodf_full.nii.gz')

    ret = script_runner.run('scil_compute_asym_odf_metrics.py', in_fodf,
                            '--not_all', '--cos_asym_map',
                            'cos_asym_map.nii.gz', '-f')
    assert ret.success
