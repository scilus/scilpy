#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_project_streamlines_to_map.py', '--help')
    assert ret.success


def test_execution_tractometry_default(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'tractometry',
                             'IFGWM_uni.trk')
    in_ref = os.path.join(get_home(), 'tractometry',
                          'mni_masked.nii.gz')
    ret = script_runner.run('scil_project_streamlines_to_map.py', in_bundle,
                            'out_def/', '--in_metrics', in_ref)

    assert ret.success


def test_execution_tractometry_wm(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'tractometry',
                             'IFGWM_uni.trk')
    in_ref = os.path.join(get_home(), 'tractometry',
                          'mni_masked.nii.gz')
    ret = script_runner.run('scil_project_streamlines_to_map.py', in_bundle,
                            'out_wm/', '--in_metrics', in_ref,
                            '--to_wm', '--from_wm')

    assert ret.success
