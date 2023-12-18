#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip', 'commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_reshape_to_reference.py', '--help')
    assert ret.success


def test_execution_others(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(get_home(), 'others', 't1_crop.nii.gz')
    in_ref = os.path.join(get_home(), 'others', 't1.nii.gz')
    ret = script_runner.run('scil_volume_reshape_to_reference.py', in_img,
                            in_ref, 't1_reshape.nii.gz',
                            '--interpolation', 'nearest')
    assert ret.success


def test_execution_4D(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(get_home(), 'commit_amico', 'dwi.nii.gz')
    in_ref = os.path.join(get_home(), 'others', 't1.nii.gz')
    ret = script_runner.run('scil_volume_reshape_to_reference.py', in_img,
                            in_ref, 'dwi_reshape.nii.gz',
                            '--interpolation', 'nearest')
    assert ret.success
