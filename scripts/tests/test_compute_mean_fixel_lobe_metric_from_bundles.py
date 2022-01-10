#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
        'scil_compute_mean_fixel_lobe_metric_from_bundles.py',
        '--help')

    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'processing',
                           'fodf_descoteaux07.nii.gz')
    in_bundles = os.path.join(get_home(), 'processing', 'tracking.trk')

    # generate bingham volume
    script_runner.run('scil_fit_bingham_to_fodf.py',
                      in_fodf, 'bingham.nii.gz',
                      '--max_lobes', '2',
                      '--at', '0.0',
                      '--rt', '0.1',
                      '--min_sep_angle', '25.',
                      '--max_fit_angle', '15.',
                      '--processes', '1')

    script_runner.run('scil_compute_lobe_specific_fodf_metrics.py',
                      'bingham.nii.gz', '--nbr_integration_steps', '10',
                      '--processes', '1', '--not_all', '--out_fd',
                      'fd.nii.gz')

    ret = script_runner.run(
        'scil_compute_mean_fixel_lobe_metric_from_bundles.py',
        in_bundles, 'bingham.nii.gz', 'fd.nii.gz',
        'fixel_mean_fd.nii.gz', '--length_weighting')

    assert ret.success
