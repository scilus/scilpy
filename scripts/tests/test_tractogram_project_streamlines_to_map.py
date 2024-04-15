#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_project_streamlines_to_map.py',
                            '--help')
    assert ret.success


def test_execution_dpp(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM_uni.trk')
    in_mni = os.path.join(SCILPY_HOME, 'tractometry', 'mni_masked.nii.gz')
    in_bundle_with_dpp = 'IFGWM_uni_with_dpp.trk'

    # Create our test data with dpp: add metrics as dpp.
    # Or get a tractogram that already as some dpp in the test data.
    script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                      in_bundle, in_bundle_with_dpp, '-f',
                      '--in_maps', in_mni, '--out_dpp_name', 'some_metric')

    # Tests with dpp.
    ret = script_runner.run('scil_tractogram_project_streamlines_to_map.py',
                            in_bundle_with_dpp, 'project_dpp_',
                            '--use_dpp', 'some_metric', '--point_by_point',
                            '--to_endpoints')
    assert ret.success

    ret = script_runner.run('scil_tractogram_project_streamlines_to_map.py',
                            in_bundle_with_dpp, 'project_mean_to_endpoints_',
                            '--use_dpp', 'some_metric', '--mean_streamline',
                            '--to_endpoints')
    assert ret.success

    ret = script_runner.run('scil_tractogram_project_streamlines_to_map.py',
                            in_bundle_with_dpp, 'project_end_to_wm',
                            '--use_dpp', 'some_metric', '--mean_endpoints',
                            '--to_wm')
    assert ret.success


def test_execution_dps(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM_uni.trk')
    in_mni = os.path.join(SCILPY_HOME, 'tractometry', 'mni_masked.nii.gz')
    in_bundle_with_dpp = 'IFGWM_uni_with_dpp.trk'
    in_bundle_with_dps = 'IFGWM_uni_with_dps.trk'

    # Create our test data with dps: add metrics as dps.
    # Or get a tractogram that already as some dps in the test data.
    script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                      in_bundle, in_bundle_with_dpp, '-f',
                      '--in_maps', in_mni, '--out_dpp_name', 'some_metric')
    script_runner.run('scil_tractogram_dpp_math.py', 'min', in_bundle_with_dpp,
                      in_bundle_with_dps, '--in_dpp_name', 'some_metric',
                      '--out_keys', 'some_metric_dps', '--mode', 'dps',
                      '--keep_all')

    # Tests with dps.
    ret = script_runner.run('scil_tractogram_project_streamlines_to_map.py',
                            in_bundle_with_dps, 'project_dps_',
                            '--use_dps', 'some_metric_dps', '--point_by_point',
                            '--to_wm')
    assert ret.success
