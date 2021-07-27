#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_bingham_fodf_metrics.py',
                            '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'processing',
                           'fodf_descoteaux07.nii.gz')
    ret = script_runner.run('scil_compute_bingham_fodf_metrics.py',
                            in_fodf,
                            '--max_lobes', '1',
                            '--at', '0.0',
                            '--rt', '0.1',
                            '--min_sep_angle', '25.',
                            '--max_fit_angle', '15.',
                            '--nbr_integration_steps', '10',
                            '--processes', '1')
    assert ret.success
