#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_dpp_math.py', '--help')
    assert ret.success


def test_execution_tractogram_point_math_mean_3D_defaults(script_runner,
                                                          monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM_uni.trk')
    in_t1 = os.path.join(SCILPY_HOME, 'tractometry', 'mni_masked.nii.gz')
    t1_on_bundle = 't1_on_streamlines.trk'

    # Create some dpp. Could have test data with dpp instead.
    script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                      in_bundle, t1_on_bundle, '--in_maps', in_t1,
                      '--out_dpp_name', 't1')

    # Test dps mode
    ret = script_runner.run('scil_tractogram_dpp_math.py',
                            'mean', t1_on_bundle, 't1_mean_on_streamlines.trk',
                            '--mode', 'dps', '--in_dpp_name', 't1',
                            '--out_keys', 't1_mean')

    assert ret.success

    # Test dpp mode
    ret = script_runner.run('scil_tractogram_dpp_math.py',
                            'mean', t1_on_bundle,
                            't1_mean_on_streamlines2.trk',
                            '--mode', 'dpp', '--in_dpp_name', 't1',
                            '--out_keys', 't1_mean')

    assert ret.success


def test_execution_tractogram_point_math_mean_4D_correlation(script_runner,
                                                             monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tracking', 'local_split_0.trk')
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    fodf_on_bundle = 'fodf_on_streamlines.trk'

    script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                      in_bundle, fodf_on_bundle,
                      '--in_maps', in_fodf, in_fodf,
                      '--out_dpp_name', 'fodf', 'fodf2')

    ret = script_runner.run('scil_tractogram_dpp_math.py',
                            'correlation', fodf_on_bundle,
                            'fodf_correlation_on_streamlines.trk',
                            '--mode', 'dps', '--endpoints_only',
                            '--in_dpp_name', 'fodf',
                            '--out_keys', 'fodf_correlation')

    assert ret.success
