#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_project_streamlines_to_map.py',
                            '--help')
    assert ret.success


def test_execution_dpp(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'tractometry', 'IFGWM_uni.trk')
    in_ref = os.path.join(get_home(), 'tractometry', 'mni_masked.nii.gz')
    in_bundle_with_dpp = 'IFGWM_uni_with_dpp.trk'

    # Create our test data with dpp: add metrics as dpp.
    # Or get a tractogram that already as some dpp in the test data.
    script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                      in_bundle, in_ref, in_bundle_with_dpp,
                      '--dpp_name', 'some_metric')

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


def test_execution_dps(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'tractometry', 'IFGWM_uni.trk')
    in_ref = os.path.join(get_home(), 'tractometry', 'mni_masked.nii.gz')
    in_bundle_with_dpp = 'IFGWM_uni_with_dpp.trk'
    in_bundle_with_dps = 'IFGWM_uni_with_dps.trk'

    # Create our test data with dps: add metrics as dpp.
    # Or get a tractogram that already as some dps in the test data.
    script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                      in_bundle, in_ref, in_bundle_with_dpp,
                      '--dpp_name', 'some_metric', '-f')
    script_runner.run('scil_tractogram_dpp_math.py', in_bundle_with_dpp,
                      in_ref, in_bundle_with_dpp, '--dpp_name', 'some_metric',
                      '-f')

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