#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_generate_gradient_sampling.py', '--help')
    assert ret.success


def test_execution_others(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_generate_gradient_sampling.py',
                            '32', '64', 'encoding.b', '--mrtrix', '--eddy',
                            '--duty', '--b0_every', '25', '--b0_end',
                            '--bvals', '800', '1200')
    assert ret.success
