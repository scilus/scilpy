#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_convert_gradients_fsl_to_mrtrix.py',
                            '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bval = os.path.join(get_home(), 'processing',
                           '1000.bval')
    in_bvec = os.path.join(get_home(), 'processing',
                           '1000.bvec')
    ret = script_runner.run('scil_convert_gradients_fsl_to_mrtrix.py',
                            in_bval, in_bvec, '1000.b')
    assert ret.success
