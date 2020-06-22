#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_crop_volume.py', '--help')
    assert ret.success


def test_execution_others(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_img = os.path.join(get_home(), 'others',
                             't1.nii.gz')
    ret = script_runner.run('scil_crop_volume.py', input_img, 't1_crop.nii.gz')
    assert ret.success
