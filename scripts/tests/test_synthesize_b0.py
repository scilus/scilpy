#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip', 'processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_synthesize_b0.py', '--help')
    assert ret.success


def test_synthesis(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_t1 = os.path.join(get_home(), 'others',
                         't1.nii.gz')
    in_b0 = os.path.join(get_home(), 'processing',
                         'b0_mean.nii.gz')
    ret = script_runner.run('scil_synthesize_b0.py',
                            in_t1, in_b0, 'b0_synthesized.nii.gz', '-v')
    assert ret.success
