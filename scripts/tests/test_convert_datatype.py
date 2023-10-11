#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tempfile

from scilpy.io.fetcher import get_home, fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['atlas.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_convert_datatype.py', '--help')
    assert ret.success


def test_execution(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_img = os.path.join(get_home(), 'atlas', 'brainstem.nii.gz')
    ret = script_runner.run('scil_image_math.py', in_img,
                            'test.nii.gz', 'uint8')
    assert ret.success
