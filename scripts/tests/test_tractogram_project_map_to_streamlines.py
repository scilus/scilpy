#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
            'scil_tractogram_project_map_to_streamlines.py', '--help')
    assert ret.success


def test_execution_3D_map(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_t1 = os.path.join(get_home(), 'tractometry', 'mni_masked.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1, 't1_on_streamlines.trk',
                            '--in_metric', in_t1,
                            '--out_dpp_name', 't1')
    assert ret.success


def test_execution_4D_map(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_rgb = os.path.join(get_home(), 'others', 'rgb.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1, 'rgb_on_streamlines.trk',
                            '--in_metric', in_rgb,
                            '--out_dpp_name', 'rgb')
    assert ret.success


def test_execution_3D_map_endpoints_only(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_t1 = os.path.join(get_home(), 'tractometry', 'mni_masked.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1,
                            't1_on_streamlines_endpoints.trk',
                            '--in_metric', in_t1,
                            '--out_dpp_name', 't1',
                            '--endpoints_only')
    assert ret.success


def test_execution_4D_map_endpoints_only(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_rgb = os.path.join(get_home(), 'others', 'rgb.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1,
                            'rgb_on_streamlines_endpoints.trk',
                            '--in_metric', in_rgb,
                            '--out_dpp_name', 'rgb',
                            '--endpoints_only')
    assert ret.success


def test_execution_3D_map_trilinear(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_t1 = os.path.join(get_home(), 'tractometry', 'mni_masked.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1,
                            't1_on_streamlines_trilinear.trk',
                            '--in_metric', in_t1,
                            '--out_dpp_name', 't1',
                            '--trilinear')
    assert ret.success
