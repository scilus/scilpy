#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip', 'tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_tractogram_extract_streamlines',
                            '--help'])
    assert ret.success

def _create_tractogram_with_dpp(script_runner):
    """
    Copied this code from test_tractogram_projects_streamlines_to_map
    ToDo: Add a tractogram with dpp in our test data.
    """
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM_uni.trk')
    in_mni = os.path.join(SCILPY_HOME, 'tractometry', 'mni_masked.nii.gz')
    in_bundle_with_dpp = 'IFGWM_uni_with_dpp.trk'

    # Create our test data with dpp: add metrics as dpp.
    # Or get a tractogram that already as some dpp in the test data.
    script_runner.run(['scil_tractogram_project_map_to_streamlines',
                       in_bundle, in_bundle_with_dpp, '-f',
                       '--in_maps', in_mni, '--out_dpp_name', 'some_metric'])


def test_from_dpp(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # Creating the data
    # Could eventually split this test into many tests, but we want to create 
    # the data only once
    _create_tractogram_with_dpp(script_runner)
    in_bundle_with_dpp = 'IFGWM_uni_with_dpp.trk'

    # From NB, top
    ret = script_runner.run(['scil_tractogram_extract_streamlines',
                             in_bundle_with_dpp, 'out_200_top.trk',
                             '--from_dpp', 'some_metric',
                             '--top', '--nb', 200])
    assert ret.success

    # From NB, center
    ret = script_runner.run(['scil_tractogram_extract_streamlines',
                             in_bundle_with_dpp, 'out_200_middle.trk',
                             '--from_dpp', 'some_metric',
                             '--center', '--nb', 200])
    assert ret.success

     # From Percent, bottom
    ret = script_runner.run(['scil_tractogram_extract_streamlines',
                             in_bundle_with_dpp, 'out_5percent_bottom.trk',
                             '--from_dpp', 'some_metric',
                             '--bottom', '--percent', 5])   

    assert ret.success

     # From mean + std, center
    ret = script_runner.run(['scil_tractogram_extract_streamlines',
                             in_bundle_with_dpp, 'out_middle_std.trk',
                             '--from_dpp', 'some_metric',
                             '--center', '--mean_std', 3])   

    assert ret.success


def from_dps(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # Creating the data. We create a file with dpp and then average it over
    # streamlines as dps.
    # Could eventually split this test into many tests, but we want to create 
    # the data only once
    _create_tractogram_with_dpp(script_runner)
    in_bundle_with_dpp = 'IFGWM_uni_with_dpp.trk'
    in_bundle_with_dps = 'IFGWM_uni_with_dps.trk'
    script_runner.run(['scil_tractogram_dpp_math', 'MEAN', in_bundle_with_dpp, 
                       in_bundle_with_dps, '--mode', 'dps',
                       '--in_dpp_name', 'some_metric', 
                       '--out_keys', 'mean_dpp'])

    # No need to retest all options. 
    # From NB, top
    ret = script_runner.run(['scil_tractogram_extract_streamlines',
                             in_bundle_with_dps, 'out_200_top.trk',
                             '--from_dps', 'mean_dpp',
                             '--top', '--nb', 200])
    assert ret.success
