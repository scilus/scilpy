#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_visualize_bingham_fit.py', '--help')
    assert ret.success


def test_silent_without_output(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    # dummy dataset (the script should raise an error before using it)
    in_dummy = os.path.join(get_home(), 'tracking', 'fodf.nii.gz')

    ret = script_runner.run('scil_visualize_bingham_fit.py', in_dummy,
                            '--silent')

    assert (not ret.success)
