#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()


##### Deprecated file but it should still be running.
##### For more exhaustive tests, see test_tractogram_math.py

def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py', '--help')
    assert ret.success


def test_execution_3D_map(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fa = os.path.join(get_home(), 'others', 'fa.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1, in_fa, 'fa_on_streamlines.trk')
    assert ret.success

def test_execution_4D_map(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_rgb = os.path.join(get_home(), 'others', 'rgb.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1, in_rgb, 'rgb_on_streamlines.trk')
    assert ret.success

def test_execution_3D_map_endpoints_only(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fa = os.path.join(get_home(), 'others', 'fa.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1, in_fa, 'fa_on_streamlines.trk',
                            '--endpoints_only')
    assert ret.success

def test_execution_4D_map_endpoints_only(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_rgb = os.path.join(get_home(), 'others', 'rgb.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1, in_rgb, 'rgb_on_streamlines.trk',
                            '--endpoints_only')
    assert ret.success

def test_execution_3D_map_dpp_name(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fa = os.path.join(get_home(), 'others', 'fa.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1, in_fa, 'fa_on_streamlines.trk',
                            '--dpp_name', 'fa')
    assert ret.success

def test_execution_3D_map_trilinear(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fa = os.path.join(get_home(), 'others', 'fa.nii.gz')
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'IFGWM_sub.trk')

    ret = script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                            in_tracto_1, in_fa, 'fa_on_streamlines.trk',
                            '--trilinear')
    assert ret.success